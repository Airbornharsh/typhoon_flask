import re


def highest_numbered_file(file_names):
    # Extract numbers from file names
    numbers = [int(re.search(r"\d+", file_name).group()) for file_name in file_names]

    # Find the highest number
    highest_number = max(numbers)

    # Construct the highest file name
    highest_file_name = f"{highest_number}.csv"

    return highest_file_name
