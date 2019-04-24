#!/usr/bin/env python

def send_mail():

    import smtplib 
    
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.MIMEImage import MIMEImage

    # me == my email address
    # you == recipient's email address
    me = "sokhosh@gmail.com"
    you = "khosh.ar@gmail.com"

    # Create message container - the correct MIME type is multipart/alternative.
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "testing"
    msg['From'] = me
    msg['To'] = you
    
    # Create the body of the message (a plain-text and an HTML version).
    text = "Hi!\nHow are you?\nHere is the link you wanted:\nhttps://www.python.org"
    html = """\
    <html>
    <head></head>
    <body>
    <p>Hi!<br>
       How are you?<br>
       Here is the <a href="https://www.python.org">link</a> you wanted.
    </p>
    </body>
    </html>
    """

    # Record the MIME types of both parts - text/plain and text/html.
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')
    part3 = MIMEText(file("strats.pdf").read())

    # Attach parts into message container.
    # According to RFC 2046, the last part of a multipart message, in this case
    # the HTML message, is best and preferred.
    msg.attach(part1)
    msg.attach(part2)
    msg.attach(part3)

    gmail_pwd=''
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login(me, gmail_pwd)
    server.sendmail(me, you, msg.as_string())
    # server.quit()
    server.close()


send_mail()
