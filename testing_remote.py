import matplotlib.pyplot as plt

print("Helloooo!!!!!!")

plt.plot([x for x in range(10)], [x * 2 for x in range(10)])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("HI")

plt.savefig('remote1.png')
plt.show()