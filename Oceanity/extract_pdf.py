from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(filename="datasets/Slide 1.pdf")
# print(elements)
for el in elements:
    print(f"{el.category}: {el.text[:100]}")

import json

output = []
for el in elements:
    entry = {
        "type": el.category,
        "text": el.text,
        "page_number": el.metadata.page_number,
        "coordinates": el.metadata.coordinates
    }
    output.append(entry)

with open("parsed_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
print("Xong")