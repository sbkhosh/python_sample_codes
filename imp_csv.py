#!/usr/bin/python3

import csv

with open('amx.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    with open('amx_new.csv', 'r') as csv_file_new:
        fieldnames = ['Date', 'Open', 'Close']

        csv_writer = csv.DictWriter(csv_file_new, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        for line in csv_reader:
            csv_writer.writerow(line)



# with open('amx.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)

#     for line in csv_reader:
#         print(line['Close'])
        


# with open('amx_new.csv', 'r') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter = '\t')

#     for line in csv_reader:
#         print(line)
        
    
    
    # with open('amx_new.csv', 'w') as csv_file_new:
    #     csv_writer = csv.writer(csv_file_new, delimiter='\t')

    

