import json
import pickle
import os
from collections import defaultdict
from collections import Counter

data = []
origin_file_path = "./eval_datas/"
for filename in os.listdir(origin_file_path):
  if filename.endswith(".json"):
    with open(origin_file_path + filename, 'r', encoding='utf-8') as f:
      file_data = json.load(f)
      data.extend(file_data)

merged_data = defaultdict(lambda: {"file_name": "", "text": "", "plot_elements": []})

for item in data:
  file_name = None
  text = None
  plot_element = "None"
  try:
    file_name = item["file_name"]
    text = item["text"]
    plot_element = item.get("plot_element", "None")
  except:
    file_name = item["data"]["file_name"]
    text = item["data"]["text"]
    result = item["annotations"][0]["result"]
    for r in result:
      if r["from_name"] == "plot_element":
        plot_element = r["value"]["choices"][0]
  
  key = (file_name, text)
  merged_data[key]["file_name"] = file_name
  merged_data[key]["text"] = text
  merged_data[key]["plot_elements"].append(plot_element)

output_data = list(merged_data.values())
print(str(len(output_data)) + ' datas loaded complete')

filtered_data = [item for item in output_data if "plot_elements" in item and "None" not in item["plot_elements"]]
filtered_len = len(filtered_data)
deleted_num = len(output_data) - filtered_len
print(str(deleted_num) + ' data was deleted.' + 'total: ' + str(filtered_len))


for idx, item in enumerate(filtered_data, start=1):
  item["id"] = idx
  plot_elements = item.get("plot_elements", [])
  if len(plot_elements) == 3:
    if plot_elements[0] == plot_elements[1] == plot_elements[2]:
      item["element_num"] = 1
    elif plot_elements[0] == plot_elements[1] or plot_elements[1] == plot_elements[2] or plot_elements[0] == plot_elements[2]:
      item["element_num"] = 2
    else:
      item["element_num"] = 3
print('complete add id and element_num')

with open('novel_data_valid.json', 'w', encoding='utf-8') as f:
  json.dump(filtered_data, f, ensure_ascii=False, indent=4)
  print('complete save json file')
with open('novel_data_valid.pkl', 'wb') as f:
  pickle.dump(filtered_data, f)
  print('complete save pkl file')
