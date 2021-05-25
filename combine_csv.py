import csv

import csv
reader = csv.reader(open("doublepp_gap_2000_copy.csv"))
reader1 = csv.reader(open("doublepp_gap_850steps_jan6.csv"))
f = open("doublepp_combined_2850.csv", "w")
writer = csv.writer(f)
next(reader1)
for row in reader:
    writer.writerow(row)
for row in reader1:
    if (float(row[1]) != "r_in"):
        writer.writerow(row)
f.close()
"""
my_file_name = "flat_combined_2000.csv" 
cleaned_file = "flat_combined_2000.csv" 
remove_words = ['r_in'] 
with open(my_file_name, 'r', newline='') as infile, \
    open(cleaned_file, 'w',newline='') as outfile: 
   writer = csv.writer(outfile) 
   for line in csv.reader(infile, delimiter='|'): 
       if not any(remove_word in line for remove_word in remove_words): 
           writer.writerow(line)

"""
