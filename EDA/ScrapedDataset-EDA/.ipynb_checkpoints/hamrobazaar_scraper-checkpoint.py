"""
Production Mobile Phone Scraper for Hamrobazaar.com
Optimized for large-scale data collection
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from datetime import datetime
import os

class HamrobazaarScraper:
    def __init__(self, headless=True):
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--log-level=3')
        options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = webdriver.Chrome(options=options)
        self.data = []
    
    def collect_listing_urls(self, base_url, max_listings=1000):
        """Collect product URLs using infinite scroll - aggressive collection for large datasets"""
        print(f"Collecting up to {max_listings} listing URLs (infinite scroll)...")
        self.driver.get(base_url)
        time.sleep(3)
        
        all_links = set()
        no_new_count = 0
        scroll_count = 0
        max_scrolls = 1000  # High limit for large datasets
        
        while len(all_links) < max_listings and scroll_count < max_scrolls:
            previous_count = len(all_links)
            
            # Get visible links
            elems = self.driver.find_elements(By.CSS_SELECTOR, 'a.newTabAnchor')
            for elem in elems:
                try:
                    href = elem.get_attribute('href')
                    if href and '/product/' in href:
                        all_links.add(href)
                except:
                    pass
            
            # Check progress
            new_count = len(all_links) - previous_count
            if new_count == 0:
                no_new_count += 1
                if no_new_count >= 10:  # Stop if no new links after 10 scrolls
                    print(f"  ✓ Reached end of listings (no new items after 10 scrolls)")
                    break
            else:
                no_new_count = 0
                if len(all_links) % 100 == 0:  # Progress every 100 links
                    print(f"  Progress: {len(all_links)} URLs collected...")
            
            # Aggressive scrolling
            scroll_count += 1
            self.driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)  # Balance speed and loading
            
            # Occasionally scroll back up to trigger more loading
            if scroll_count % 20 == 0:
                self.driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(0.5)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.5)
        
        urls = list(all_links)[:max_listings]
        print(f"\n✓ Collected {len(urls)} URLs after {scroll_count} scrolls\n")
        return urls
    
    def extract_listing_data(self, url):
        """Extract all data from a single listing"""
        self.driver.get(url)
        time.sleep(2)
        
        data = {}
        
        # Title
        for sel in ['h1.title--normal', 'h2.product-title', 'h1', '[class*="title"]']:
            try:
                data['title'] = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                if data['title']:
                    break
            except:
                continue
        
        # Price
        for sel in ['.regularPrice', '.price', '[class*="price"]']:
            try:
                data['price'] = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                if data['price']:
                    break
            except:
                continue
        
        # Extract all features
        try:
            features = self.driver.find_elements(By.CSS_SELECTOR, '.feature__item')
            for f in features:
                try:
                    label = f.find_element(By.CSS_SELECTOR, '.label span').text.strip().lower()
                    try:
                        value = f.find_element(By.CSS_SELECTOR, '.label__desc').text.strip()
                    except:
                        value = f.find_element(By.CSS_SELECTOR, '.label__desc span').text.strip()
                    
                    # Map features to data fields
                    if 'location' in label:
                        data['location'] = value
                    elif 'condition' in label:
                        data['condition'] = value
                    elif 'negotiable' in label:
                        data['negotiable'] = value
                    elif 'warranty' in label:
                        data['warranty'] = value
                    elif 'brand' in label:
                        data['brand'] = value
                    elif 'model' in label:
                        data['model'] = value
                    elif 'storage' in label or 'internal storage' in label:
                        data['storage'] = value
                    elif 'ram' in label:
                        data['ram'] = value
                    elif 'battery' in label:
                        data['battery'] = value
                    elif 'back camera' in label:
                        data['back_camera'] = value
                    elif 'front camera' in label:
                        data['front_camera'] = value
                    elif 'camera' in label and 'back' not in label and 'front' not in label:
                        data['camera'] = value
                    elif 'screen' in label or 'display' in label:
                        data['screen_size'] = value
                    elif 'color' in label or 'colour' in label:
                        data['color'] = value
                    elif 'sim' in label:
                        data['sim_slot'] = value
                    elif 'processor' in label or 'chipset' in label:
                        data['processor'] = value
                    elif 'used for' in label:
                        data['used_for'] = value
                    elif 'ownership' in label:
                        data['ownership_document'] = value
                except:
                    pass
        except:
            pass
        
        # Seller name
        for sel in ['.username-fullname', '.seller-name', '[class*="seller"]']:
            try:
                data['seller_name'] = self.driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                if data['seller_name']:
                    break
            except:
                continue
        
        # Description
        for sel in ['.tab-content .ad--informations .ad--desc p', 
                    '.ad--informations .ad--desc p',
                    '.ad--desc p',
                    '.tab-content .ad--desc p']:
            try:
                desc_elem = self.driver.find_element(By.CSS_SELECTOR, sel)
                desc_text = desc_elem.text.strip()
                if desc_text:
                    data['description'] = desc_text
                    break
            except:
                continue
        
        return data
    
    def scrape(self, url, max_listings=100, output_file='hamrobazaar_mobile_data.csv', resume=False):
        """
        Main scraping method
        
        Args:
            url: Category URL to scrape
            max_listings: Maximum number of listings to scrape
            output_file: CSV filename for output
            resume: If True, skip already scraped listings
        """
        start_time = time.time()
        
        # Load existing data if resuming
        scraped_count = 0
        if resume and os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            scraped_count = len(existing_df)
            self.data = existing_df.to_dict('records')
            print(f"📂 Resuming: {scraped_count} listings already scraped\n")
        
        # Collect URLs
        urls = self.collect_listing_urls(url, max_listings)
        
        # Scrape each listing
        print(f"Scraping {len(urls)} listings...")
        print("="*70)
        
        for i, listing_url in enumerate(urls, 1):
            try:
                print(f"[{i}/{len(urls)}]", end=" ")
                data = self.extract_listing_data(listing_url)
                
                if data and data.get('title'):
                    self.data.append(data)
                    print(f"✓ {data['title'][:50]}")
                    
                    # Save progress every 25 listings for large datasets
                    save_interval = 25 if max_listings >= 500 else 10
                    if i % save_interval == 0:
                        self._save_progress(output_file)
                        elapsed = time.time() - start_time
                        avg_time = elapsed / i
                        remaining = (len(urls) - i) * avg_time
                        print(f"    💾 Progress saved ({len(self.data)} listings) | ⏱️  ETA: {remaining/60:.1f} mins")
                else:
                    print("✗ Failed (no data)")
            except Exception as e:
                print(f"✗ Error: {str(e)[:40]}")
        
        # Final save
        self.driver.quit()
        df = pd.DataFrame(self.data)
        df.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print("="*70)
        print(f"\n✅ Scraping Complete!")
        print(f"   Total listings: {len(self.data)}")
        print(f"   Time taken: {elapsed/60:.1f} minutes ({elapsed/len(urls):.1f}s per listing)")
        print(f"   Output file: {output_file}")
        
        return df
    
    def _save_progress(self, filename):
        """Save current progress to CSV"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Configuration
    TARGET_URL = "https://hamrobazaar.com/category/mobile-phone-handsets/0618E1EF-00AF-4EAC-8E07-4978A2C7BB5E/2A2C15D9-9A08-4E77-A292-F8AD803C2490"
    MAX_LISTINGS = 1000  # Will collect as many as available via infinite scroll
    OUTPUT_FILE = "hamrobazaar_mobile_1000.csv"
    
    print("="*70)
    print("HAMROBAZAAR MOBILE PHONE SCRAPER")
    print("="*70)
    print(f"Target: {MAX_LISTINGS} listings")
    print(f"Output: {OUTPUT_FILE}")
    print("="*70 + "\n")
    
    # Run scraper
    scraper = HamrobazaarScraper(headless=True)
    df = scraper.scrape(TARGET_URL, max_listings=MAX_LISTINGS, output_file=OUTPUT_FILE)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Fields: {', '.join(df.columns)}")
