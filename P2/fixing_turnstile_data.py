import csv
import pprint

def fix_turnstile_data(filenames):
    '''
    Filenames is a list of MTA Subway turnstile text files. A link to an example
    MTA Subway turnstile text file can be seen at the URL below:
    http://web.mta.info/developers/data/nyct/turnstile/turnstile_110507.txt
    
    As you can see, there are numerous data points included in each row of the
    a MTA Subway turnstile text file. 

    You want to write a function that will update each row in the text
    file so there is only one entry per row. A few examples below:
    A002,R051,02-00-00,05-28-11,00:00:00,REGULAR,003178521,001100739
    A002,R051,02-00-00,05-28-11,04:00:00,REGULAR,003178541,001100746
    A002,R051,02-00-00,05-28-11,08:00:00,REGULAR,003178559,001100775
    
    Write the updates to a different text file in the format of "updated_" + filename.
    For example:
        1) if you read in a text file called "turnstile_110521.txt"
        2) you should write the updated data to "updated_turnstile_110521.txt"

    The order of the fields should be preserved. Remember to read through the 
    Instructor Notes below for more details on the task. 
    
    In addition, here is a CSV reader/writer introductory tutorial:
    http://goo.gl/HBbvyy
    
    You can see a sample of the turnstile text file that's passed into this function
    and the the corresponding updated file in the links below:
    
    Sample input file:
    https://www.dropbox.com/s/mpin5zv4hgrx244/turnstile_110528.txt
    Sample updated file:
    https://www.dropbox.com/s/074xbgio4c39b7h/solution_turnstile_110528.txt
    '''

    for file in filenames:
        # create file input object f_in to work with in_data.csv file
        f_in = open(file, 'r')
        # create file output object f_out to write to the new out_data.csv
        f_out = open('updated_' + file, 'w')
        # creater csv readers and writers based on our file objects
        reader_in  = csv.reader(f_in, delimiter=',')
        writer_out = csv.writer(f_out, delimiter=',')

        for row in reader_in:
            location = row[:3]
            date_times = list(chunks(row[3:], 5))

            for dt in date_times: 
                full_row = location + dt
                full_row = [x.strip() for x in full_row]
                writer_out = csv.writer(f_out)
                writer_out.writerow(full_row)

        f_in.close()
        f_out.close()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def test_data(file):
    PATH = '/Users/joshRpowell/Dropbox/Udacity/DataAnalyst/P2/'

    # create file input object f_in to work with in_data.csv file
    f_in = open(PATH + file, 'r')
    # create file output object f_out to write to the new out_data.csv
    f_out = open(PATH + 'updated_' + file, 'w')
    # creater csv readers and writers based on our file objects
    reader_in = csv.reader(f_in, delimiter=',')
    writer_out = csv.writer(f_out, delimiter=',')

    for i in range(2):
        rows = reader_in.next()
        location = rows[:3]
        # print location
        date_times = list(chunks(reader_in.next()[3:], 5))
        # pprint.pprint(date_times)
        for dt in date_times: 
            full_row = location + dt
            # print full_row
            full_row = [x.strip() for x in full_row]
            print full_row
            writer_out = csv.writer(f_out)
            writer_out.writerow(full_row)

test_data('turnstile_110528.txt')
