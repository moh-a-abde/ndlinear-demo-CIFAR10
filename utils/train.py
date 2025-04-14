import time
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report


def train_and_evaluate(baseline, ndmodel, train_loader, test_loader, device, epochs=30):
    import torch.optim as optim

    optimizer_b = optim.Adam(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_n = optim.Adam(ndmodel.parameters(), lr=1.5e-3, weight_decay=5e-5)
    scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode='max', factor=0.5, patience=2, threshold=0.005)
    scheduler_n = optim.lr_scheduler.ReduceLROnPlateau(optimizer_n, mode='max', factor=0.5, patience=2, threshold=0.005)

    baseline_metrics = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'inference_time': [], 'lr': []}
    ndlinear_metrics = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'inference_time': [], 'lr': []}

    base_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    nd_params = sum(p.numel() for p in ndmodel.parameters() if p.requires_grad)

    print(f"=== PARAMETER COUNT ===")
    print(f"Baseline param count:  {base_params:,}")
    print(f"NdLinear param count:  {nd_params:,}  ({(1 - nd_params / base_params) * 100:.1f}% reduction)")

    best_acc_b, best_acc_n = 0, 0
    patience_counter_b, patience_counter_n = 0, 0
    early_stopping_patience = 5

    for epoch in range(1, epochs + 1):
        baseline.train()
        ndmodel.train()
        bl_loss = nd_loss = bl_acc = nd_acc = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total += batch_size

            optimizer_b.zero_grad()
            out = baseline(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer_b.step()

            bl_loss += loss.item() * batch_size
            bl_acc += (out.argmax(dim=1) == labels).sum().item()

            optimizer_n.zero_grad()
            out = ndmodel(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            optimizer_n.step()

            nd_loss += loss.item() * batch_size
            nd_acc += (out.argmax(dim=1) == labels).sum().item()

        bl_loss /= total
        nd_loss /= total
        bl_acc = 100.0 * bl_acc / total
        nd_acc = 100.0 * nd_acc / total

        baseline.eval()
        ndmodel.eval()
        bl_test_acc = nd_test_acc = 0
        bl_preds = bl_labels = nd_preds = nd_labels = []
        bl_infer_time = nd_infer_time = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                batch_size = labels.size(0)
                test_total += batch_size

                start = time.time()
                out = baseline(images)
                bl_infer_time += time.time() - start
                preds = out.argmax(dim=1)
                bl_test_acc += (preds == labels).sum().item()
                bl_preds.extend(preds.cpu().tolist())
                bl_labels.extend(labels.cpu().tolist())

                start = time.time()
                out = ndmodel(images)
                nd_infer_time += time.time() - start
                preds = out.argmax(dim=1)
                nd_test_acc += (preds == labels).sum().item()
                nd_preds.extend(preds.cpu().tolist())
                nd_labels.extend(labels.cpu().tolist())

        bl_test_acc = 100.0 * bl_test_acc / test_total
        nd_test_acc = 100.0 * nd_test_acc / test_total

        scheduler_b.step(bl_test_acc)
        scheduler_n.step(nd_test_acc)

        baseline_metrics['train_loss'].append(bl_loss)
        baseline_metrics['train_acc'].append(bl_acc)
        baseline_metrics['test_acc'].append(bl_test_acc)
        baseline_metrics['inference_time'].append(bl_infer_time)
        baseline_metrics['lr'].append(optimizer_b.param_groups[0]['lr'])

        ndlinear_metrics['train_loss'].append(nd_loss)
        ndlinear_metrics['train_acc'].append(nd_acc)
        ndlinear_metrics['test_acc'].append(nd_test_acc)
        ndlinear_metrics['inference_time'].append(nd_infer_time)
        ndlinear_metrics['lr'].append(optimizer_n.param_groups[0]['lr'])

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Baseline: loss={bl_loss:.4f}, train_acc={bl_acc:.2f}%, test_acc={bl_test_acc:.2f}%, "
              f"lr={optimizer_b.param_groups[0]['lr']:.1e} | "
              f"NdLinear: loss={nd_loss:.4f}, train_acc={nd_acc:.2f}%, test_acc={nd_test_acc:.2f}%, "
              f"lr={optimizer_n.param_groups[0]['lr']:.1e}")

        if bl_test_acc > best_acc_b:
            best_acc_b = bl_test_acc
            patience_counter_b = 0
            torch.save(baseline.state_dict(), "results/best_baseline.pth")
        else:
            patience_counter_b += 1

        if nd_test_acc > best_acc_n:
            best_acc_n = nd_test_acc
            patience_counter_n = 0
            torch.save(ndmodel.state_dict(), "results/best_ndlinear.pth")
        else:
            patience_counter_n += 1

        if patience_counter_b >= early_stopping_patience and patience_counter_n >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    bl_report = classification_report(bl_labels, bl_preds, digits=4)
    nd_report = classification_report(nd_labels, nd_preds, digits=4)

    return baseline_metrics, ndlinear_metrics, base_params, nd_params, bl_report, nd_report