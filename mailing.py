#!/usr/bin/python3

import concurrent.futures
import requests
import time
import re
import bs4
import requests
import email, smtplib, ssl
from io import BytesIO
from reportlab.pdfgen import canvas
from django.http import HttpResponse
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from bs4 import BeautifulSoup

def evaluate_job(job_url):
    try:
        job_html = requests.request('GET', job_url, timeout = 10)
    except:
        return 0
    
    job_soup = bs4.BeautifulSoup(job_html.content, 'lxml')
    soup_body = job_soup('body')[0]

    python_count = soup_body.text.count('Python') + soup_body.text.count('python')
    sql_count = soup_body.text.count('SQL') + soup_body.text.count('sql')
    cpp_count = soup_body.text.count('C++') + soup_body.text.count('c++')
    skill_count = python_count + sql_count + cpp_count
    print('C++ count: {0}, Python count: {1}, SQL count: {2}'.format(cpp_count, python_count, sql_count))
    return(skill_count)
    
def extract_job_data_from_indeed(base_url):
    response = requests.get(base_url)
    soup = bs4.BeautifulSoup(response.content, 'lxml')
    
    tags = soup.find_all('div', {'data-tn-component' : "organicJob"})
    companies_list = [x.span.text for x in tags]
    attrs_list = [x.h2.a.attrs for x in tags]
    dates = [x.find_all('span', {'class':'date'}) for x in tags]
    
    # update attributes dictionaries with company name and date posted
    [attrs_list[i].update({'company': companies_list[i].strip()}) for i, x in enumerate(attrs_list)]
    [attrs_list[i].update({'date posted': dates[i][0].text.strip()}) for i, x in enumerate(attrs_list)]
    return(attrs_list)

def find_new_jobs(days_ago_limit = 14, starting_page = 0, pages_limit = 20, old_jobs_limit = 5, location = 'London', query = 'quant'):    
    extra_interest_companies = ['Citadel', 'AQR', 'Rokos']
    query_formatted = re.sub(' ', '+', query)
    location_formatted = re.sub(' ', '+', location)
    indeed_url = 'http://www.indeed.co.uk/jobs?q={0}&l={1}&sort=date&start='.format(query_formatted, location_formatted)
    old_jobs_counter = 0
    new_jobs_list = []
    
    for i in xrange(starting_page, starting_page + pages_limit):
        if old_jobs_counter >= old_jobs_limit:
            break
        
        print('URL: {0}'.format(indeed_url + str(i*10)), '\n')

        # extract job data from Indeed page
        attrs_list = extract_job_data_from_indeed(indeed_url + str(i*10))
        
        # loop through each job, breaking out if we're past the old jobs limit
        for j in xrange(0, len(attrs_list)): 
            if old_jobs_counter >= old_jobs_limit:
                break

            href = attrs_list[j]['href']
            title = attrs_list[j]['title']
            company = attrs_list[j]['company']
            date_posted = attrs_list[j]['date posted']
            
            # if posting date is beyond the limit, add to the counter and skip
            try:
                if int(date_posted[0]) >= days_ago_limit:
                    print('Adding to old_jobs_counter')
                    old_jobs_counter+= 1
                    continue
            except:
                pass

            print('{0}, {1}, {2}'.format(repr(company), repr(title), repr(date_posted)))

            # evaluate the job
            evaluation = evaluate_job('https://www.indeed.co.uk' + href)
            
            if evaluation >= 1 or company.lower() in extra_interest_companies:
                new_jobs_list.append('{0}, {1}, {2}'.format(company, title, 'http://indeed.co.uk' + href))
                
            print('\n')
            time.sleep(1)
            
    new_jobs_string = '\n\n'.join(new_jobs_list)
    return new_jobs_string

def send_gmail(from_addr = "sbkhosh@gmail.com", to_addr ="sbkhosh@gmail.com",
               location = 'Dubai',
               subject = 'job update', text = None):
    
    message = 'Subject: {0}\n\nJobs in: {1}\n\n{2}'.format(subject, location, text)

    # login information
    password = "ewskaukcgicoytij"
    
    # send the message
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(from_addr,password)
    server.sendmail(from_addr, to_addr, message)
    server.quit()
    print('Email sent')

if __name__ == '__main__':
    start_page = 0
    page_limit = 2
    location = 'London'
    data_scientist_jobs = find_new_jobs(query = 'quant', starting_page = start_page,
                                        location = location, pages_limit = page_limit, days_ago_limit = 1, old_jobs_limit = 5)
    send_gmail(text = data_scientist_jobs, location = location)
