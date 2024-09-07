import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json


# Load gmail password
with open('key.json', 'r') as file:
    config = json.load(file)

pw = config['email_password']

def send_email(recipient, subject, body):
    port = 465
    smtp_server = "smtp.gmail.com"
    sender_email = "calvinz168@gmail.com"
    password = pw

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient

    msg['Subject'] = subject
    body = MIMEText(body)
    msg.attach(body)

    server = smtplib.SMTP_SSL(smtp_server, port)

    server.login(sender_email, password)
    server.sendmail(sender_email, recipient, msg.as_string())
    server.quit()

