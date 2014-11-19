#! /usr/bin/python
# -*- coding: utf-8 -*-

# Convert digits in transcripts to Chinese expression
# usage: python num_date_char.py -i input_transcripts -o output_transcripts
# Support: 1. up to 8 digit for integer number
#          2. for numbers such as x.x or x%, need to normalize '.', and '%' 
#             to Chinese character before using this script
#          3. more than 8 digits will be treated as digit sequence
#          4. xxxx年xx月xx日

import argparse,re
__author__ = 'Minhua Wu'
 
parser = argparse.ArgumentParser(description='Convert digit numbers to Chinese characters')
parser.add_argument('-i','--input', help='Input file name (Containing digits)',required=True)
parser.add_argument('-o','--output',help='Output file name (Containing only Chinese characters)', required=True)
args = parser.parse_args()

digit_dict = {"0":"零","1":"一","2":"二","3":"三","4":"四","5":"五","6":"六","7":"七","8":"八","9":"九"}


def within_two_digit_convert(Num_w2):
    Chinese_w2 = ""
    if len(Num_w2) == 1:
        Chinese_w2 = digit_dict[Num_w2]
    elif Num_w2 == "10":
        Chinese_w2 = " 十 "
    elif Num_w2[0] == "1":
        Chinese_w2 = " 十 " + digit_dict[Num_w2[1]]
    elif Num_w2[1] == "0":
        Chinese_w2 = digit_dict[Num_w2[0]]+ " 十 "
    else:
        Chinese_w2 = digit_dict[Num_w2[0]]+ " 十 " +digit_dict[Num_w2[1]]
    return Chinese_w2

def only_two_digit_convert(Num_o2):
    Chinese_o2 = ""
    if Num_o2 == "10":
        Chinese_o2 = " 十 "
    elif Num_o2[0] == "1":
        Chinese_o2 = " 十 " + digit_dict[Num_o2[1]]
    elif Num_o2[1] == "0":
        Chinese_o2 = digit_dict[Num_o2[0]]+ " 十 "
    else:
        Chinese_o2 = digit_dict[Num_o2[0]]+ " 十 " +digit_dict[Num_o2[1]]
    return Chinese_o2

def two_digit_convert(Num2):
    Chinese2 = ""
    if Num2 == "00":
        Chinese2 = ""
    elif Num2[0] == "0":
        Chinese2 = " 零 " +  digit_dict[Num2[1]]
    elif Num2[1] == "0":
        Chinese2 = digit_dict[Num2[0]] + " 十 "
    else:
        Chinese2 = digit_dict[Num2[0]]+ " 十 " +digit_dict[Num2[1]]
    return Chinese2

def three_digit_convert(Num3):
    Chinese3 = ""
    if Num3 == "000":
        Chinese3 =""
    elif Num3[0] == "0" and Num3[1] == "0":
        Chinese3 =" 零 " + digit_dict[Num3[2]] 
    elif Num3[0] == "0":
        Chinese3 = " 零 " + two_digit_convert(Num3[1:])
    else:
        Chinese3 = digit_dict[Num3[0]]+ " 百 " + two_digit_convert(Num3[1:])
    return Chinese3

def four_digit_convert(Num4):
    Chinese4 = ""
    if Num4 == "0000":
        Chinese4 = ""
    else:
        Chinese4 = digit_dict[Num4[0]] + " 千 " + three_digit_convert(Num4[1:])
    return Chinese4
        
def more_digit_convert(Num_more):
    Chinese_more = ""
    if len(Num_more) == 5:
        Chinese_more = digit_dict[Num_more[0]] + " 万 " + four_digit_convert(Num_more[1:])
    elif len(Num_more) == 6:
        Chinese_more = only_two_digit_convert(Num_more[0:2]) + " 万 "+ four_digit_convert(Num_more[2:])
    elif len(Num_more) == 7:
        Chinese_more = three_digit_convert(Num_more[0:3]) + " 万 "+ four_digit_convert(Num_more[3:])
    elif len(Num_more) == 8:
        Chinese_more = four_digit_convert(Num_more[0:4]) + " 万 "+ four_digit_convert(Num_more[4:])
    return Chinese_more

def digit_seq_convert(Num_seq):
    Chinese_seq = ""
    for digit in Num_seq:
        Chinese_seq = Chinese_seq + " " + digit_dict[digit]
    return Chinese_seq

# integer within 4 digits  
def integer_convert(integer):
    Chinese_int = ""    
    if len(integer) == 1:
        Chinese_int = digit_dict[integer]
    elif len(integer) == 2:
        Chinese_int = only_two_digit_convert(integer)
    elif len(integer) == 3:
        Chinese_int = three_digit_convert(integer)
    elif len(integer) == 4:
        Chinese_int = four_digit_convert(integer)
    elif len(integer) >= 5 and len(integer) <= 8:
        Chinese_int = more_digit_convert(integer)
    elif len(integer) >= 9:
        Chinese_int = digit_seq_convert(integer) 
    return Chinese_int

fin = open(args.input,"r")
fout = open(args.output, "w")

convert_chars = ""
for sentence in fin:
    l=sentence.split()
    utt_id = l[0]
    line = ''.join(l[1:])

    patterns = []    
    patterns = re.findall(r'[12]\d\d\d年\d+月[123]?\d日',line)
    for item in patterns:
        year = re.match(r'(\d+)年(\d+)月(\d+)日',item).group(1)
        month = re.match(r'(\d+)年(\d+)月(\d+)日',item).group(2)
        day = re.match(r'(\d+)年(\d+)月(\d+)日',item).group(3)
        convert_chars = digit_seq_convert(year) + " 年 " \
                        + within_two_digit_convert(month) + " 月 " \
                        + within_two_digit_convert(day) + " 日 "
        line = line.replace(item,convert_chars)
   
    patterns = []
    patterns = re.findall(r'[12]\d\d\d年\d+月',line)
    for item in patterns:
        year = re.match(r'(\d+)年(\d+)月',item).group(1)
        month = re.match(r'(\d+)年(\d+)月',item).group(2)
        convert_chars = digit_seq_convert(year) + " 年 " \
                        + within_two_digit_convert(month) + " 月 "
        line = line.replace(item,convert_chars)
     
    patterns = []
    patterns = re.findall(r'\d+月[123]?\d日',line)
    for item in patterns:
        month = re.match(r'(\d+)月(\d+)日',item).group(1)
        day = re.match(r'(\d+)月(\d+)日',item).group(2)
        convert_chars = within_two_digit_convert(month) + " 月 " \
                        + within_two_digit_convert(day) + " 日 "
        line = line.replace(item,convert_chars)
    
    patterns = []
    patterns = re.findall(r'[12]\d\d\d年',line)
    for item in patterns:
        year = re.match(r'(\d+)年',item).group(1)
        convert_chars = digit_seq_convert(year) + " 年 "
        line = line.replace(item,convert_chars)

    patterns = []
    patterns = re.findall(r'\d+月',line)
    for item in patterns:
        month = re.match(r'(\d+)月',item).group(1)
        convert_chars = within_two_digit_convert(month) + " 月 " 
        line = line.replace(item,convert_chars)

    patterns = []
    patterns = re.findall(r'[123]?\d日',line)
    for item in patterns:
        day = re.match(r'(\d+)日',item).group(1)
        convert_chars = within_two_digit_convert(day) + " 日 " 
        line = line.replace(item,convert_chars)

    patterns = []
    patterns = re.findall(r'\d+分之\d+',line)
    for item in patterns:
        denominator = re.match(r'(\d+)分之(\d+)',item).group(1)
        numerator = re.match(r'(\d+)分之(\d+)',item).group(2)
        convert_chars = integer_convert(denominator) + " 分 之 " + integer_convert(numerator)
        line = line.replace(item,convert_chars)

    patterns = []
    patterns = re.findall(r'\d+点\d+',line)
    for item in patterns:
        left = re.match(r'(\d+)点(\d+)',item).group(1)
        right = re.match(r'(\d+)点(\d+)',item).group(2)
        convert_chars = integer_convert(left) + " 点 " + digit_seq_convert(right)
        line = line.replace(item,convert_chars)

    patterns = []
    patterns = re.findall(r'[^\d]0\d*',line)
    for item in patterns:
        seq = re.match(r'[^\d](0\d*)',item).group(1)
        convert_chars = digit_seq_convert(seq)
        line = line.replace(seq,convert_chars,1)

    patterns = []
    patterns = re.findall(r'\d+',line)
    for item in patterns:
        convert_chars = integer_convert(item)
        line = line.replace(item,convert_chars,1)

    line = ''.join(line.split())
    fout.write(utt_id+" "+line+"\n") 
    
fin.close()
fout.close()
