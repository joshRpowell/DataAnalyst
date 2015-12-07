#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
"""
Your task is to explore the data a bit more.
The first task is a fun one - find out how many unique users
have contributed to the map in this particular area!

The function process_map should return a set of unique user IDs ("uid")
"""

def get_user(element):
	if 'uid' in element.attrib.keys():
		user = element.attrib['uid']

		return user


def process_map(filename):
    users = set()
    user_list = []
    for _, element in ET.iterparse(filename):
	  	if 'uid' in element.attrib.keys():
			user = element.attrib['uid']
  			user_list.append(user)

    users = set(user_list)
    return users


def test():

    users = process_map('example.osm')
    pprint.pprint(users)
    assert len(users) == 6



if __name__ == "__main__":
    test()