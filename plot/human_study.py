import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

values_a = [30, 55, 35]
values_b = [75, 35, 60]
plt.figure(figsize=(8, 4))

y_major_locator = MultipleLocator(10)
method1 = ['Clean', 'Baseline', 'Ours']
method2 = ['Clean vs \n Clean', 'Clean vs \n Baseline', 'Clean vs \n Ours']
plt.subplots_adjust(wspace=0.2, hspace=0)  # Adjust subgraph distance

ax1 = plt.subplot(121)
a = plt.bar(range(len(values_a)), values_a, color=['r', 'g', 'b'], tick_label=method1)

# ax is entity of two axises
ax = plt.gca()
# Set the major scale of the y-axis to a multiple of 10
ax.yaxis.set_major_locator(y_major_locator)
# Set coordinate axis range
plt.ylim((0, 80))
plt.xlabel("(a)")
plt.ylabel("Percentage of Noisy Examples(%)")
# add data
plt.bar_label(a, label_type='edge')

# second figure
plt.subplot(122)
b = plt.bar(range(len(values_b)), values_b, color=['r', 'g', 'b'], tick_label=method2)
# ax is entity of two axises
ax = plt.gca()
# Set the major scale of the y-axis to a multiple of 10
ax.yaxis.set_major_locator(y_major_locator)
# Set coordinate axis range
plt.ylim((0, 80))
plt.xlabel("(b)")
plt.ylabel("Percentage of Identical Examples(%)")
# add data
plt.bar_label(b, label_type='edge')
plt.show()
