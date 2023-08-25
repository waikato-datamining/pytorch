import argparse
import csv
import os

parser = argparse.ArgumentParser(
    description="Converts text files to CSV.",
    prog="gpt_text2csv",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", help="The text file to convert", default=None, required=True)
parser.add_argument("-o", "--output", help="The CSV file to save the converted data to; if omitted, simply changes the extension of the input file to .csv and saves it under that file name.", default=None, required=False)
parsed = parser.parse_args()

# determine input/output files
input_file = parsed.input
if parsed.output is None:
    output_file = os.path.splitext(input_file)[0] + ".csv"
else:
    output_file = parsed.output

# convert data
with open(input_file, encoding='utf-8') as fp:
    all_text = fp.read()
with open(output_file, mode='w', encoding='utf-8') as fp:
    fieldnames = ['text']
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'text': all_text})
