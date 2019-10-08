#!/usr/bin/python3

from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# specifies the path to the chromedriver.exe
driver = webdriver.Chrome('/home/skhosh/Downloads/chromedriver')

# # driver.get method() will navigate to a page given by the URL address
# driver.get('https://www.linkedin.com')

# # locate email form by_class_name
# username = driver.find_element_by_class_name('login-email')

# # send_keys() to simulate key strokes
# username.send_keys('sbkhosh@gmail.com')

# # locate password form by_class_name
# password = driver.find_element_by_class_name('login-password')

# # send_keys() to simulate key strokes
# password.send_keys('A3hura312')

# # locate submit button by_class_name
# log_in_button = driver.find_element_by_class_name('login submit-button')

# # locate submit button by_class_id
# log_in_button = driver.find_element_by_class_id('login-submit')

# # locate submit button by_xpath
# log_in_button = driver.find_element_by_xpath('//*[@type="submit"]')

# # .click() to mimic button click
# log_in_button.click()


search_query = 'site:linkedin.com/in/ AND "python developer" AND "London"'
file_name = 'results_file.csv'

driver.get('https:www.google.com')
sleep(3)

search_query = driver.find_element_by_name('q')
search_query.send_keys(search_query)
sleep(0.5)

# search_query.send_keys(Keys.RETURN)
# sleep(3)

# linkedin_urls = driver.find_elements_by_class_name('iUh30')
# linkedin_urls = [url.text for url in linkedin_urls]
# sleep(0.5)

driver.quit()
