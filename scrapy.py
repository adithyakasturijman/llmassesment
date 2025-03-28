import os
import time
import json
import csv
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai


def load_env():
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set GEMINI_API_KEY in your .env file")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('models/gemini-1.5-flash-latest')


def setup_selenium():
    options = Options()
    options.headless = False
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def fetch_content(driver, url):
    driver.get(url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup


def llm_model(model, prevdata, data, keywords):
    prompt = f"""From this data {data}
            check the {prevdata} to know what all are answered before, if a question is answered just continue if its not aswer the unanswered question
            analyze and try to extract all the answers to the below questions 
            1.What is the company's mission statement or core values?(search for about or ethics page or vision or philosophy)
            2.What products or services does the company offer?(search for products or mobility or services or technology)
            3.When was the company founded, and who were the founders?(search for leadership page or company information or history)
            4.Where is the company's headquarters located?(search for contact page)
            5.Who are the key executives or leadership team members?(search for leadership page)
            6.Has the company received any notable awards or recognitions?
            Provide the output as a valid JSON object embedded in a list:
                "question": "Actua Question"
                "status": "completed/notcompleted",
                "msg": "answer text or null",
                "link": ["list of full URLs if not completed by using {keywords} the URLs find by this keywords may have the answer"]
            Ensure that the response is only valid JSON {{}} without any ```json and ```.
            """

    try:
        generation_config = {'temperature': 0.3, 'max_output_tokens': 1024}

        response = model.generate_content(prompt, generation_config=generation_config)
        extracted_info = response.text.strip()
        extracted_info = extracted_info.replace('```json', '')
        try:
            return json.loads(extracted_info.replace('`', ''))
        except json.JSONDecodeError:
            print("Invalid JSON output from Gemini. Raw response:", extracted_info)
            return None
    except Exception as e:
        return {"error": f"LLM extraction error: {e}"}


from urllib.parse import urljoin

def rescrap(driver, res, prevdata, model, keywords, base_url, max_retries=5):
    visited_links = set() 
    retries = {}

    while True: 
        unanswered_links = []
        
        for item in res:
            if item["status"] != "completed" and item["link"]:
                for link in item["link"]:
                    absolute_link = base_url+link if not link.startswith("http") else link
                    
                    if absolute_link not in visited_links:
                        unanswered_links.append(absolute_link)
                        retries[absolute_link] = retries.get(absolute_link, 0) + 1
        
        if not unanswered_links:
            print("All questions answered or no more links to check. Stopping recursion.")
            break  

        prevdata = res  
        
        for link in unanswered_links:
            if retries[link] > max_retries:
                print(f"Max retries reached for {link}. Skipping further attempts.")
                continue 

            print(f"Fetching content from: {link}")
            try:
                soup = fetch_content(driver, link)  
                res = llm_model(model, prevdata, soup, keywords) 
                visited_links.add(link)  
                prevdata = res 
                print(res)
            except Exception as e:
                print(f"Error fetching {link}: {e}")

    return res





def main():
    model = load_env()
    driver = setup_selenium()
    COMPANY_KEYWORDS = {
    "https://www.apple.com": [
        'business', 'newsroom', 'compliance', 'contact'
    ],
    "https://www.toyota-global.com": [
        'executives', 'global-vision', 'company', 'financial-results', 'profile', 'contact'
    ],
    "https://www.jpmorganchase.com": [
        'leadership', 'awards-and-recognition', 'our-history', 'suppliers', 'business-principles', 'contact'
    ],
    "https://www.pfizer.com": [
        'product-list', 'executives', 'history', 'purpose', 'global-impact', 'contact'
    ],
    "https://www.thewaltdisneycompany.com": [
        'our-businesses', 'news', 'social-impact', 'about', 'contact'
    ],
    "https://www.shell.com": [
        'who-we-are', 'our-values', 'our-history', 'news-and-insights', 'contact'
    ],
    "https://www.nestle.com": [
        'management', 'purpose-values', 'nestle-company-history', 'global-addresses', 'global-goals', 'contact'
    ],
    "https://www.siemens.com": [
        'products.html', 'management.html', 'contact', 'telegraphy-and-telex.html', 'system-and-method-for-robotic-picking.html', 'technology-to-transform-the-everyday.html'
    ],
    "https://www.samsung.com/in/": [
        'company-info', 'business-area', 'brand-identity', 'environment', 'contact'
    ],
    "https://www.nike.com": [
        'about', 'investors', 'sustainability', 'contact'
    ]
    }
    url = ["https://www.apple.com","https://www.toyota-global.com","https://www.pfizer.com","https://www.thewaltdisneycompany.com","https://www.shell.com","https://www.nestle.com","https://www.siemens.com","https://www.samsung.com","https://www.nike.com"]
    for i in url:
        keywords = COMPANY_KEYWORDS[i]
        soup = fetch_content(driver, i)
        prevdata = 'null'
        res = llm_model(model, prevdata, soup, keywords)
        print(res)
        res = rescrap(driver, res, prevdata, model, keywords, i)
        csv_file = "newoutput.csv"
        write_header = not os.path.exists(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as output_file:
            fieldnames = ["original_link", "question", "status", "msg", "link"]
            dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

            if write_header: 
                dict_writer.writeheader()

            for row in res:
                row["original_link"] = i 
                dict_writer.writerow(row)


    driver.quit()

if __name__ == "__main__":
    main()
