#program that determines the identity of an individual via matching their DNA via a databse

import sys
import csv

people_database = []  # stores small.csv & big.csv
dna_str_library = []  # stores STRs present in small.csv & big.csv
STR_result = {}  # stores result of maximum STR count


def dnastrlibrary(database):
    for i in range(len(database)):
        for key, values in database[i].items():
            if key not in "name" and key not in dna_str_library:
                dna_str_library.append(key)


# use this to create a dict of STR results

def repeat_number(short_tandem_repeat, file):
    counter = 0
    max_repeats = 0
    x = len(short_tandem_repeat)

    for index in range(len(file)):

        if file[index: index + x] == short_tandem_repeat and file[index + x: index + (2 * x)] != short_tandem_repeat:
            counter += 1
            if counter > max_repeats:
                max_repeats = counter
                counter = 0
            else:
                counter = 0

        elif file[index: index + x] == short_tandem_repeat and file[index + x: index + (2 * x)] == short_tandem_repeat:
            counter += 1

    STR_result[short_tandem_repeat] = str(max_repeats)
    

# matches STR result to person in database

def match(database, result):
    matches = []

    for person in database:
        if STR_result.items() <= person.items():
            matches.append(person["name"])

    if not matches:
        print("No match")
    else:
        print(matches[0])


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python dna.py [csvfile] [textfile]")

    with open(sys.argv[1], "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for person in reader:
            people_database.append(person)

    dnastrlibrary(people_database)

    with open(sys.argv[2], "r") as dnatxtfile:
        dna_string = dnatxtfile.read()

        for STR in dna_str_library:
            repeat_number(STR, dna_string)
            
    
main()
match(people_database, STR_result)
