import re
import requests

# 2017 combined release page
url = "https://www150.statcan.gc.ca/n1/pub/62m0004x/62m0004x2017001-eng.htm"
r = requests.get(url, timeout=60)
r.raise_for_status()
links = re.findall(r'href=["\']([^"\']+)["\']', r.text, re.I)
print("=== 2017 page links (filtered) ===")
for l in sorted(set(links)):
    ll = l.lower()
    if any(x in ll for x in [".zip", ".csv", "eft", "download", "data", "shs", "62m"]):
        print(l)

# archived pub pages
print("\n=== archived pub pages ===")
for yr in range(1997, 2010):
    for suffix in [f"62m0004x{yr}001-eng.htm", f"62m0004x{yr:04d}001-eng.htm"]:
        u = f"https://www150.statcan.gc.ca/n1/pub/62m0004x/{suffix}"
        rr = requests.get(u, timeout=30)
        if rr.status_code == 200:
            print(yr, "OK", u)
            zips = [x for x in re.findall(r'href=["\']([^"\']+)["\']', rr.text, re.I) if ".zip" in x.lower()]
            for z in zips:
                print("  ", z)

# try open data API
print("\n=== catalogue search ===")
cat = requests.get("https://www150.statcan.gc.ca/n1/en/catalogue/62M0004X", timeout=60)
for m in re.findall(r'href="(/n1/[^"]+)"', cat.text):
    if "62m0004" in m.lower() or "more" in m.lower():
        print(m)
