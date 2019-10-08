#!/usr/bin/python3

def send_email():
            import smtplib

            gmail_user = ""
            gmail_pwd = ""
            FROM = ''
            TO = ['']
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

            gmail_user = ""
            gmail_pwd = ""
            FROM = ''
            TO = ['']
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

