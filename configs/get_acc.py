import json
import numpy as np
import csv

# 初始化结果列表
results = []
headers = ['Seed', 'Top-1 Acc (%)', 'Top-5 Acc (%)', 'mAP (%)']

for s in range(10):
    try:
        with open(f'./seed{s}/test_results.json') as f:
            res = json.load(f)[0]

            # 提取并计算结果
            acc1 = np.round(res['test_top1_acc'], 4) * 100
            acc5 = np.round(res['test_top5_acc'], 4) * 100
            map_val = np.round(res['mAP'], 4) * 100

            # 添加到结果列表
            results.append({
                'Seed': s,
                'Top-1 Acc (%)': acc1,
                'Top-5 Acc (%)': acc5,
                'mAP (%)': map_val
            })

            # 打印当前结果
            print(f"Seed {s}: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%, mAP={map_val:.2f}%")

            if s == 4:
                print('\n')

    except FileNotFoundError:
        print(f"⚠️ 警告: seed{s} 的结果文件未找到，跳过该种子")
    except Exception as e:
        print(f"⚠️ 错误: 处理 seed{s} 时出错: {str(e)}")

# 计算平均值
if results:
    acc1_avg = np.mean([r['Top-1 Acc (%)'] for r in results])
    acc5_avg = np.mean([r['Top-5 Acc (%)'] for r in results])
    map_avg = np.mean([r['mAP (%)'] for r in results])

    # 添加平均值行
    results.append({
        'Seed': 'Average',
        'Top-1 Acc (%)': acc1_avg,
        'Top-5 Acc (%)': acc5_avg,
        'mAP (%)': map_avg
    })

    # 写入CSV文件
    with open('results_summary.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ 结果已保存到 results_summary.csv")
    print(f"平均值: Top-1={acc1_avg:.2f}%, Top-5={acc5_avg:.2f}%, mAP={map_avg:.2f}%")
else:
    print("⚠️ 警告: 未找到任何结果数据")
