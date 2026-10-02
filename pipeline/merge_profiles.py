import json
import glob

profiles = []
for path in sorted(glob.glob("profiles/*.json")):
    profiles.append(json.load(open(path)))

json.dump(profiles, open("profiles.json", "w"), indent=2)
print(f"Merged {len(profiles)} profiles -> profiles.json")
for p in profiles:
    print(" -", p.get("name"))