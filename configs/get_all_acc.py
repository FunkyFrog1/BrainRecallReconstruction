import json
import numpy as np
import csv

# 初始化结果列表
results = []
headers = ['Subject', 'Seed', 'Top-1 Acc (%)', 'Top-5 Acc (%)', 'mAP (%)']

for sub in range(2):
    for s in range(10):
        try:
            with open(f'./sub-{sub + 1:02d}/seed{s}/test_results.json') as f:
                res = json.load(f)[0]

                # 提取并计算结果
                acc1 = np.round(res['test_top1_acc'], 4) * 100
                acc5 = np.round(res['test_top5_acc'], 4) * 100
                map_val = np.round(res['mAP'], 4) * 100

                # 添加到结果列表，包含被试编号
                results.append({
                    'Subject': sub + 1,
                    'Seed': s,
                    'Top-1 Acc (%)': acc1,
                    'Top-5 Acc (%)': acc5,
                    'mAP (%)': map_val
                })

                # 打印当前结果
                print(f"Subject {sub + 1:02d}, Seed {s}: Top-1={acc1:.2f}%, Top-5={acc5:.2f}%, mAP={map_val:.2f}%")

                if s == 4:
                    print('\n')

        except FileNotFoundError:
            print(f"⚠️ 警告: sub-{sub + 1:02d}/seed{s} 的结果文件未找到，跳过")
        except Exception as e:
            print(f"⚠️ 错误: 处理 sub-{sub + 1:02d}/seed{s} 时出错: {str(e)}")

# 计算每个被试的平均值
subjects = set(r['Subject'] for r in results)
for subject in sorted(subjects):
    subject_results = [r for r in results if r['Subject'] == subject]

    acc1_avg = np.mean([r['Top-1 Acc (%)'] for r in subject_results])
    acc5_avg = np.mean([r['Top-5 Acc (%)'] for r in subject_results])
    map_avg = np.mean([r['mAP (%)'] for r in subject_results])

    # 添加被试平均值行
    results.append({
        'Subject': subject,
        'Seed': 'Average',
        'Top-1 Acc (%)': acc1_avg,
        'Top-5 Acc (%)': acc5_avg,
        'mAP (%)': map_avg
    })

    print(f"Subject {subject:02d} 平均值: Top-1={acc1_avg:.2f}%, Top-5={acc5_avg:.2f}%, mAP={map_avg:.2f}%")

# 计算总体平均值
if results:
    # 只计算原始结果（不包括平均值行）
    original_results = [r for r in results if r['Seed'] != 'Average']

    overall_acc1 = np.mean([r['Top-1 Acc (%)'] for r in original_results])
    overall_acc5 = np.mean([r['Top-5 Acc (%)'] for r in original_results])
    overall_map = np.mean([r['mAP (%)'] for r in original_results])

    # 添加总体平均值行
    results.append({
        'Subject': 'Overall',
        'Seed': 'Average',
        'Top-1 Acc (%)': overall_acc1,
        'Top-5 Acc (%)': overall_acc5,
        'mAP (%)': overall_map
    })

    print(f"\n总体平均值: Top-1={overall_acc1:.2f}%, Top-5={overall_acc5:.2f}%, mAP={overall_map:.2f}%")

# 写入单个CSV文件
if results:
    with open('all_subjects_results_summary.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ 所有被试结果已保存到 all_subjects_results_summary.csv")
else:
    print("⚠️ 警告: 未找到任何结果数据")
