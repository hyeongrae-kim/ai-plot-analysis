import json
from collections import defaultdict

input_path = './train_data/'
input_files = ['plot_element_1.json', 'plot_element_2.json']
output_file = 'novel_plot_total.json'
output_data = []

for input_file in input_files:
  with open(input_path + input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
  for item in data:
    file_name = item.get('data', {}).get('file_name', '').strip()
    chunk_id = item.get('data', {}).get('chunk_id', '')
    text = item.get('data', {}).get('text', '')
    annotations = item.get('annotations', [])

    tag = ''
    sentence = ''
    weight = ''
    for annotation in annotations:
      results = annotation.get('result', [])
      for result in results:
        value = result.get('value', {})
        from_name = result.get('from_name', '')
        if from_name == 'plot_element':
          choices = value.get('choices', [])
          if choices:
            tag = choices[0]
        elif from_name == 'sentence':
          sentence_text = value.get('text', '')
          if sentence_text:
            sentence = sentence_text
        elif from_name == 'degree':
          choices = value.get('choices', [])
          if choices:
            weight = choices[0]

    if tag == "" or tag =="None":
      continue
    output_entry = {
      "file_name": file_name,
      "chunk_id": chunk_id,
      "text": text,
      "sentence": sentence,
      "tag": tag,
      "weight": weight
    }
    output_data.append(output_entry)

file_name_to_entries = defaultdict(list)
for entry in output_data:
  file_name = entry['file_name'].strip()
  file_name_to_entries[file_name].append(entry)

for entries in file_name_to_entries.values():
  total_chunk = len(entries)
  for idx, entry in enumerate(entries):
    entry['total_chunk'] = total_chunk
    position_value = (idx / total_chunk) * 10
    entry['position'] = int(position_value)

print("total length: " + str(len(output_data)))
with open(output_file, 'w', encoding='utf-8') as f:
  json.dump(output_data, f, ensure_ascii=False, indent=4)
print("save complete!!")
