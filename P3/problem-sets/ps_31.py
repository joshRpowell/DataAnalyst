#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In this problem set you work with cities infobox data, audit it, come up with a
cleaning idea and then clean it up. In the first exercise we want you to audit
the datatypes that can be found in some particular fields in the dataset.
The possible types of values can be:
- NoneType if the value is a string "NULL" or an empty string ""
- list, if the value starts with "{"
- int, if the value can be cast to int
- float, if the value can be cast to float, but CANNOT be cast to int.
   For example, '3.23e+07' should be considered a float because it can be cast
   as float but int('3.23e+07') will throw a ValueError
- 'str', for all other values

The audit_file function should return a dictionary containing fieldnames and a
SET of the types that can be found in the field. e.g.
{"field1: set([float, int, str]),
 "field2: set([str]),
  ....
}

All the data initially is a string, so you have to do some checks on the values
first.
"""
import codecs
import csv
import json
import pprint

CITIES = '/Users/jpowell/Dropbox/Udacity/DataAnalyst/P3/problem-sets/cities.csv'

FIELDS = ["name", "timeZone_label", "utcOffset", "homepage", "governmentType_label", "isPartOf_label", "areaCode", "populationTotal",
          "elevation", "maximumElevation", "minimumElevation", "populationDensity", "wgs84_pos#lat", "wgs84_pos#long",
          "areaLand", "areaMetro", "areaUrban"]

def audit_file(filename, fields):
    fieldtypes = {}

    with open (filename,'r') as f:
        data = list(csv.DictReader(f))
        for i in fields:
            fieldtypes[i] = []
            for row in data[3:]:
              if row[i] == "NULL" or "":
                fieldtypes[i].append(type(None))
              elif row[i].startswith('{'):
                fieldtypes[i].append(type([]))
              elif is_int(row[i]) == 'int':
                fieldtypes[i].append(type(1))
              elif is_float(row[i]) == 'float':
                fieldtypes[i].append(type(1.1))
              else:
                fieldtypes[i].append(type("a"))
            fieldtypes[i] = set(fieldtypes[i])

    # print fieldtypes
    return fieldtypes

def is_float(x):
    try:
        float(x)
        return "float"
    except ValueError:
        return 'Value'

def is_int(x):
    try:
        int(x)
        return "int"
    except ValueError:
        return 'Value'

def test():
    fieldtypes = audit_file(CITIES, FIELDS)

    pprint.pprint(fieldtypes)

    assert fieldtypes["areaLand"] == set([type(1.1), type([]), type(None)])
    assert fieldtypes['areaMetro'] == set([type(1.1), type(None)])

if __name__ == "__main__":
    test()

