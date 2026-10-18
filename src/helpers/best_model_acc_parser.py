import re
import sys

def parse_training_log(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    # Track best and running average accuracy per class
    best = {}   # class_name -> (best_acc, epoch)
    sums = {}   # class_name -> [total_acc, count]

    current_epoch = None
    for line in text.splitlines():
        epoch_match = re.match(r'Epoch (\d+)/\d+:', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue

        class_match = re.match(r'\s+Class (\w[\w\s]*\w): loss=([\d.]+), acc=([\d.]+)', line)
        if class_match and current_epoch is not None:
            name = class_match.group(1)
            acc = float(class_match.group(3))
            if name not in best or acc > best[name][0]:
                best[name] = (acc, current_epoch)
            if name not in sums:
                sums[name] = [0.0, 0]
            sums[name][0] += acc
            sums[name][1] += 1

    # Print results
    print(f"{'Class':<25} {'Best Acc':>10} {'Epoch':>7} {'Avg Acc':>10}")
    print("-" * 56)
    for name, (acc, epoch) in sorted(best.items(), key=lambda x: x[1][0], reverse=True):
        avg_acc = sums[name][0] / sums[name][1]
        print(f"{name:<25} {acc:>10.4f} {epoch:>7} {avg_acc:>10.4f}")

    # Overall averages
    overall_best_avg = sum(v[0] for v in best.values()) / len(best) if best else 0
    overall_avg = sum(s[0] / s[1] for s in sums.values()) / len(sums) if sums else 0
    print("-" * 56)
    print(f"{'Average':<25} {overall_best_avg:>10.4f} {'':>7} {overall_avg:>10.4f}")

if __name__ == '__main__':
    parse_training_log(sys.argv[1])

#python best_model_acc_parser.py "C:\Users\dk412\Desktop\GAHypN Training Plots\TRAIN_v10.2Train_epoch500_mpc3000_7541355.out"
