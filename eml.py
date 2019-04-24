#!/usr/bin/python

def send_email():
            import smtplib

            gmail_user = "sokhosh@gmail.com"
            gmail_pwd = "a3hura3&&"
            FROM = 'sokhosh@gmail.com'
            TO = ['khosh.ar@gmail.com']
            SUBJECT = "Testing sending using gmail"
            TEXT = "Testing sending mail using gmail servers"

            # Prepare actual message
            message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
            """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
            try:
                #server = smtplib.SMTP(SERVER) 
                server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
                server.ehlo()
                server.starttls()
                server.login(gmail_user, gmail_pwd)
                server.sendmail(FROM, TO, message)
                #server.quit()
                server.close()
                print 'successfully sent the mail'
            except:
                print "failed to send mail"


def send_email_test():
            import smtplib
            from email.mime.text import MIMEText

            gmail_user = "sokhosh@gmail.com"
            gmail_pwd = "a3hura3&&"
            FROM = 'sokhosh@gmail.com'
            TO = ['khosh.ar@gmail.com']
            SUBJECT = "Testing sending using gmail"

            # Prepare actual message
            message = """\From: %s\nTo: %s\nSubject: %s\n\n
            """ % (FROM, ", ".join(TO), SUBJECT)
            try:
                fp = open(textfile, 'rb')
                txt = MIMEText(fp.read())
                fp.close()
                message.attach(txt)
                # msg.attach(MIMEImage(file("strats.pdf").read()))
                #server = smtplib.SMTP(SERVER) 
                server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
                server.ehlo()
                server.starttls()
                server.login(gmail_user, gmail_pwd)
                server.sendmail(FROM, TO, message)
                #server.quit()
                server.close()
                print 'successfully sent the mail'
            except:
                print "failed to send mail"

# def send_email():
#             import smtplib
#             from email.mime.text import MIMEText

#             gmail_user = "sokhosh@gmail.com"
#             gmail_pwd = "a3hura3&&"
#             FROM = 'sokhosh@gmail.com'
#             TO = ['khosh.ar@gmail.com']
#             SUBJECT = "Testing sending using gmail"
#             TEXT = "Testing sending mail using gmail servers"

#             # Prepare actual message
#             msg = """\From: %s\nTo: %s\nSubject: %s\n\n %s
#             """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
#             try:
#                 # txt = MIMEMultipart('alt')
#                 # fp = open('textfile', 'rb')
#                 # txt.attach(MIMEText(fp.read()))
#                 # # msg = MIMEMultipart('mixed')
#                 # fp.close()
#                 server = smtplib.SMTP("smtp.gmail.com", 587)
#                 server.ehlo()
#                 server.starttls()
#                 server.login(gmail_user, gmail_pwd)
#                 server.sendmail(FROM, TO, msg.as_string())
#                 server.close()
#                 print 'successfully sent the mail'
#             except:
#                 print "failed to send mail"

send_email_test()

# Open a plain text file for reading.  For this example, assume that
# # the text file contains only ASCII characters.
# fp = open('textfile', 'rb')
# # Create a text/plain message
# msg = MIMEText(fp.read())
# # msg.attach(MIMEImage(file("strats.pdf").read())
# fp.close()

# me='sokhosh@gmail.com'
# you='khosh.ar@gmail.com'
# msg['Subject'] = 'information request'
# msg['From'] = me
# msg['To'] = you

# # Send the message via our own SMTP server, but don't include the
# # envelope header.
# server = smtplib.SMTP('smtp.gmail.com:587')
# server.ehlo()
# server.starttls()
# server.sendmail(me, [you], msg.as_string())
# server.quit()

