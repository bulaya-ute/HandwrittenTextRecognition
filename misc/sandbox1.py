def sort_rectangles(rectangles):
    centers = []
    for i, rect in enumerate(rectangles):
        x, y, w, h = rect[:4]
        centers.append((x + w // 2, y + h // 2, i))

    sorted_centers = sorted(centers, key=lambda item: (item[1]+1)**(item[0]+1))
    sorted_rects = [rectangles[sc[2]] for sc in sorted_centers]
    return sorted_rects


rects = [
    (0, 0, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1),  (1, 0, 1, 1)
]

print(sort_rectangles(rects))
