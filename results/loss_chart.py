import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['figure.dpi'] = 150

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
coat_length_train_loss = [
    0.3048, 0.3163, 0.3067, 0.2980, 0.2913,
    0.2856, 0.2801, 0.2748, 0.2696, 0.2654
]
collar_design_train_loss = [
    0.2750, 0.2204, 0.1928, 0.1578, 0.1274,
    0.1042, 0.0796, 0.0648, 0.0570, 0.0552
]
lapel_design_train_loss = [
    0.3719, 0.2898, 0.2352, 0.1713, 0.1284,
    0.1044, 0.0687, 0.0594, 0.0634, 0.0460
]
neck_design_train_loss = [
    0.3558, 0.2710, 0.2174, 0.1549, 0.1152,
    0.0929, 0.0806, 0.0616, 0.0598, 0.0524
]
neckline_design_train_loss = [
    0.1846, 0.1462, 0.1232, 0.1028, 0.0841,
    0.0738, 0.0612, 0.0531, 0.0486, 0.0472
]
pant_length_train_loss = [
    0.2825, 0.2295, 0.2026, 0.1793, 0.1566,
    0.1403, 0.1191, 0.1050, 0.0950, 0.0837
]
skirt_length_train_loss = [
    0.2873, 0.2353, 0.2064, 0.1759, 0.1513,
    0.1268, 0.1071, 0.0977, 0.0869, 0.0682
]
sleeve_length_train_loss = [
    0.2662, 0.2252, 0.2036, 0.1838, 0.1652,
    0.1503, 0.1324, 0.1166, 0.1039, 0.0870
]

plt.title('Training Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(epochs, coat_length_train_loss, marker='o', markersize=3)
plt.plot(epochs, collar_design_train_loss, marker='o', markersize=3)
plt.plot(epochs, lapel_design_train_loss, marker='o', markersize=3)
plt.plot(epochs, neck_design_train_loss, marker='o', markersize=3)
plt.plot(epochs, neckline_design_train_loss, marker='o', markersize=3)
plt.plot(epochs, pant_length_train_loss, marker='o', markersize=3)
plt.plot(epochs, skirt_length_train_loss, marker='o', markersize=3)
plt.plot(epochs, sleeve_length_train_loss, marker='o', markersize=3)

for a, b in zip(epochs, coat_length_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, collar_design_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, lapel_design_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, neck_design_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, neckline_design_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, pant_length_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, skirt_length_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, sleeve_length_train_loss):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)

plt.legend([
    'coat length', 'collar design', 'lapel design', 'neck design',
    'neckline design', 'pant length', 'skirt length', 'sleeve length'
])

plt.show()
