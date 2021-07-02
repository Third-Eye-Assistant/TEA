import smtplib
import urllib.request

def email(rec_email=None, text="Hello, It's Third Eye here...", sub='Third Eye'):
	if '@gmail.com' not in rec_email: return
	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.starttls()
	s.login("tiufyp@gmail.com", "fyp@2021") # eg, abc@gmail.com (email) and ****(pass)
	message = 'Subject: {}\n\n{}'.format(sub, text)
	s.sendmail("senderEmail", rec_email, message)
	print("Sent")
	s.quit()

if __name__ == '__main__':
    email("subhamkundu486@gmail.com")