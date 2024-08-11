from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

keywords = ["ellipse shape images","rectangle shape images", "rectangle with rounded corner shape images"]

for kw in keywords:
    response().download(kw,500)
