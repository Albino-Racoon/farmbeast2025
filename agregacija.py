import sys
import csv
import math
import os
from collections import deque

# Evklidska razdalja med dvema točkama
def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def find_numeric_columns(header, first_row):
    numeric_cols = []
    for i, key in enumerate(header):
        try:
            float(first_row[key])
            numeric_cols.append(key)
        except Exception:
            continue
        if len(numeric_cols) == 2:
            break
    if len(numeric_cols) < 2:
        raise ValueError("CSV mora imeti vsaj dva številska stolpca!")
    return numeric_cols

def load_points(csv_path):
    points = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return []
        header = reader.fieldnames
        colx, coly = find_numeric_columns(header, rows[0])
        for row in rows:
            try:
                x = float(row[colx])
                y = float(row[coly])
                points.append((x, y))
            except Exception:
                continue
    return points

def are_close(a, b, threshold=0.75):
    return abs(a[0] - b[0]) < threshold and abs(a[1] - b[1]) < threshold

def group_points(points, threshold=0.75):
    n = len(points)
    visited = [False]*n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        group = []
        queue = deque([i])
        visited[i] = True
        while queue:
            idx = queue.popleft()
            group.append(points[idx])
            for j in range(n):
                if not visited[j] and are_close(points[idx], points[j], threshold):
                    visited[j] = True
                    queue.append(j)
        groups.append(group)
    return groups

def avg_and_round(groups):
    results = []
    for group in groups:
        avg_x = sum(x for x, _ in group) / len(group)
        avg_y = sum(y for _, y in group) / len(group)
        results.append((math.floor(avg_x), math.floor(avg_y)))
    return results

def main():
    if len(sys.argv) < 2:
        print("Uporaba: python agregacija.py pot/do/csv [pot/do/drugi.csv]")
        sys.exit(1)
    csv_path1 = sys.argv[1]
    if not os.path.isfile(csv_path1):
        print(f"Datoteka {csv_path1} ne obstaja!")
        sys.exit(1)
    points = load_points(csv_path1)
    if len(sys.argv) >= 3:
        csv_path2 = sys.argv[2]
        if not os.path.isfile(csv_path2):
            print(f"Datoteka {csv_path2} ne obstaja!")
            sys.exit(1)
        points += load_points(csv_path2)
    if not points:
        print("CSV ne vsebuje uporabnih podatkov!")
        sys.exit(1)
    groups = group_points(points, threshold=0.75)
    results = avg_and_round(groups)
    out_path = os.path.splitext(csv_path1)[0] + "_agregirano.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in results:
            writer.writerow([x, y])
    print("Agregirane točke (zaokrožene navzdol):")
    for x, y in results:
        print(f"x={x}, y={y}")
    print(f"Shranjeno v: {out_path}")

if __name__ == "__main__":
    main() 