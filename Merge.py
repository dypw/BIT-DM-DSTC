import json

with open('eval_data/dailydialog_grade_transformer_generator-addrank.json', 'r') as f:
    gene = json.load(f)

with open('eval_data/dailydialog_grade_transformer_ranker-addrank.json', 'r') as f:
    ranker = json.load(f)

new_data = {}

for key in gene.keys():
    new_data[key] = gene[key] + ranker[key]

with open('eval_data/ft_daily.json', 'w') as f:
    json.dump(new_data, f)
