import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['figure.dpi'] = 150

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
coat_length_train_p = [
    0.5262, 0.6535, 0.6839, 0.7295, 0.7333,
    0.7283, 0.7501, 0.7397, 0.7496, 0.7556
]
collar_design_train_p = [
    0.7724, 0.8357, 0.8643, 0.8978, 0.9208,
    0.9385, 0.9584, 0.9675, 0.9693, 0.9692
]
lapel_design_train_p = [
    0.7162, 0.8071, 0.8551, 0.9097, 0.9402,
    0.9503, 0.9722, 0.9735, 0.9705, 0.9815
]
neck_design_train_p = [
    0.7117, 0.8087, 0.8658, 0.9160, 0.9426,
    0.9575, 0.9609, 0.9710, 0.9726, 0.9762
]
neckline_design_train_p = [
    0.7762, 0.8521, 0.8913, 0.9161, 0.9416,
    0.9462, 0.9603, 0.9650, 0.9704, 0.9689
]
pant_length_train_p = [
    0.7829, 0.8306, 0.8514, 0.8731, 0.8911,
    0.9057, 0.9210, 0.9297, 0.9385, 0.9445
]
skirt_length_train_p = [
    0.7773, 0.8279, 0.8690, 0.8973, 0.9142,
    0.9310, 0.9439, 0.9474, 0.9530, 0.9662
]
sleeve_length_train_p = [
    0.7338, 0.7812, 0.8040, 0.8304, 0.8497,
    0.8703, 0.8893, 0.9091, 0.9204, 0.9404
]

plt.title('Precision')
plt.xlabel('epoch')
plt.ylabel('p')
plt.plot(epochs, coat_length_train_p, marker='o', markersize=3)
plt.plot(epochs, collar_design_train_p, marker='o', markersize=3)
plt.plot(epochs, lapel_design_train_p, marker='o', markersize=3)
plt.plot(epochs, neck_design_train_p, marker='o', markersize=3)
plt.plot(epochs, neckline_design_train_p, marker='o', markersize=3)
plt.plot(epochs, pant_length_train_p, marker='o', markersize=3)
plt.plot(epochs, skirt_length_train_p, marker='o', markersize=3)
plt.plot(epochs, sleeve_length_train_p, marker='o', markersize=3)

for a, b in zip(epochs, coat_length_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, collar_design_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, lapel_design_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, neck_design_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, neckline_design_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, pant_length_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, skirt_length_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
for a, b in zip(epochs, sleeve_length_train_p):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=8)

plt.legend([
    'coat length', 'collar design', 'lapel design', 'neck design',
    'neckline design', 'pant length', 'skirt length', 'sleeve length'
])

plt.show()
