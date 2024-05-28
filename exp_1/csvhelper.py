import csv

def write_to_csv(data, file_path):
    """
    Write data to a CSV file

    Parameters:
    - data: A list of lists containing data
    - file_path: indicates the path of the CSV file
    """
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

    print(f'Data has been written to {file_path}')

# sample
example_data = [
    ['Name', 'Age', 'City'],
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'San Francisco'],
    ['Charlie', 22, 'Los Angeles']
]

# Specify the CSV file path
example_file_path = 'example.csv'

# Call the encapsulated method
write_to_csv(example_data, example_file_path)
