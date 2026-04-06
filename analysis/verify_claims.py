import pandas as pd

# 1. Check hard-paragraph cost savings calculation
hard = pd.read_csv('reports/benchmark_results.csv')
google_cost = hard[hard['name'] == 'google_vision']['total_cost'].values[0]
routing_cost = hard[hard['name'] == 'intelligent_routing']['total_cost'].values[0]
savings_pct = (google_cost - routing_cost) / google_cost * 100
print(f'Hard-paragraph cost savings vs Google: {savings_pct:.1f}%')
print(f'  Google total: ${google_cost:.2f}, Routing: ${routing_cost:.2f}')

# 2. Check quick-benchmark disaggregated values  
easy = pd.read_csv('reports/benchmark_results_quick_easy.csv')
medium = pd.read_csv('reports/benchmark_results_quick_medium.csv')
hard_q = pd.read_csv('reports/benchmark_results_quick_hard.csv')

print(f'\nQuick benchmarks - CRNN values:')
print(f'  Easy: {easy[easy["name"] == "custom_crnn"]["mean_cer"].values[0]:.4f} (report shows 0.4667)')
print(f'  Medium: {medium[medium["name"] == "custom_crnn"]["mean_cer"].values[0]:.4f} (report shows 0.0164)')
print(f'  Hard: {hard_q[hard_q["name"] == "custom_crnn"]["mean_cer"].values[0]:.4f} (report shows 0.0735)')

# 3. Check hard-paragraph full benchmark
print(f'\nHard-paragraph full benchmark:')
print(f'  Tesseract CER: {hard[hard["name"] == "tesseract"]["mean_cer"].values[0]:.4f} (report shows 0.976)')
print(f'  CRNN CER: {hard[hard["name"] == "custom_crnn"]["mean_cer"].values[0]:.4f} (report shows 0.448)')
print(f'  TrOCR CER: {hard[hard["name"] == "trocr"]["mean_cer"].values[0]:.4f} (report shows 1.696)')

# 4. Check classifier evaluation  
with open('reports/classifier_evaluation.txt') as f:
    eval_content = f.read()
    if '99.8%' in eval_content or '99.8' in eval_content:
        print('Checkpoint: Classifier easy accuracy 99.8% found in evaluation')
    if '0.0%' in eval_content or '0% accuracy' in eval_content:
        print('Checkpoint: Classifier medium accuracy 0% found in evaluation')
    if '54.6%' in eval_content:
        print('Checkpoint: Classifier hard accuracy 54.6% found in evaluation')
