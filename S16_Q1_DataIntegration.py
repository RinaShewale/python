# S16_Q1_DataIntegration.py


import pandas as pd

student_details = pd.DataFrame({
    'Student_ID': [101, 102, 103, 104],
    'Name': ['Riya', 'Amit', 'Neha', 'Rahul'],
    'Gender': ['Female', 'Male', 'Female', 'Male'],
    'City': ['Pune', 'Mumbai', 'Nashik', 'Nagpur']
})

academic_records = pd.DataFrame({
    'Student_ID': [101, 102, 103, 104],
    'Course': ['BCA', 'BCA', 'BCA', 'BCA'],
    'Marks': [85, 78, 92, 75],
    'Grade': ['A', 'B', 'A+', 'B']
})

print("Student Details:")
print(student_details)

print("\nAcademic Records:")
print(academic_records)

integrated_data = pd.merge(
    student_details,
    academic_records,
    on='Student_ID'
)

print("\nIntegrated Data:")
print(integrated_data)




# S16_Q2_CityVisualization.py


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

data = pd.DataFrame({
    'City': ['Pune', 'Mumbai', 'Nashik', 'Nagpur', 'Delhi'],
    'Population': [7000000, 20000000, 2000000, 3000000, 19000000],
    'GDP': [100, 310, 40, 60, 290],
    'Literacy_Rate': [88, 90, 82, 86, 89]
})

print("City Data:")
print(data)

# Treemap
plt.figure(figsize=(10, 5))

total_population = data['Population'].sum()
left = 0

for i in range(len(data)):
    width = data['Population'][i] / total_population

    rectangle = Rectangle(
        (left, 0),
        width,
        1,
        edgecolor='black',
        facecolor='skyblue'
    )

    plt.gca().add_patch(rectangle)

    plt.text(
        left + width / 2,
        0.5,
        data['City'][i],
        ha='center',
        va='center'
    )

    left = left + width

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title("City-wise Population Treemap")
plt.show()


# 3D Scatter Plot
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    data['Population'],
    data['GDP'],
    data['Literacy_Rate'],
    s=100
)

ax.set_xlabel("Population")
ax.set_ylabel("GDP")
ax.set_zlabel("Literacy Rate")
ax.set_title("3D Scatter Plot of Cities")

plt.show()