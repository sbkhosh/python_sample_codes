#!/usr/bin/env python
from smtplib import SMTP
from smtplib import SMTPException
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEImage import MIMEImage
import sys
 
EMAIL_SUBJECT = "Email from Python script with attachment."
EMAIL_FROM = 'sokhosh@gmail.com'
EMAIL_RECEIVER = 'khosh.ar@gmail.com'
GMAIL_SMTP = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587
TEXT_SUBTYPE = "plain"
 
def listToStr(lst):
    """This method makes comma separated list item string"""
    return ','.join(lst)
 
def send_email(content, pswd):
    """This method sends an email"""
   
    #Create the email.
    msg = MIMEMultipart()
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_RECEIVER
    body = MIMEMultipart('alternative')
    body.attach(MIMEText(content, TEXT_SUBTYPE ))
    #Attach the message
    msg.attach(body)
    #Attach a text file
    # msg.attach(MIMEText(file("code4reference.txt").read()))
    #Attach a picture or pdf.
    msg.attach(MIMEImage(file("strats.pdf").read()))
     
    try:
      smtpObj = SMTP(GMAIL_SMTP, GMAIL_SMTP_PORT)
      #Identify yourself to GMAIL ESMTP server.
      smtpObj.ehlo()
      #Put SMTP connection in TLS mode and call ehlo again.
      smtpObj.starttls()
      smtpObj.ehlo()
      #Login to service
      smtpObj.login(user=EMAIL_FROM, password=pswd)
      #Send email
      smtpObj.sendmail(EMAIL_FROM, EMAIL_RECEIVER, msg.as_string())
      #close connection and session.
      smtpObj.quit()
    except SMTPException as error:
      print "Error: unable to send email :  {err}".format(err=error)
 
def main(pswd):
    """This is a simple main() function which demonstrate sending of email using smtplib."""
    send_email("Test email", pswd);
 
if __name__ == "__main__":
    """If this script is run as stand alone then call main() function."""
    if len(sys.argv) == 2:
        main(sys.argv[1]);
    else:
        print "Please provide the password"
