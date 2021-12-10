about:
	@echo Testing shortcuts

ssd-wta-playroom:
	python -m stereomatch.single_image_app test-data/middleburry/playroom/im0.png test-data/middleburry/playroom/im1.png 128 ssd-wta-playroom.pgm

ssd-wta-teddy:
	python -m stereomatch.single_image_app\
		test-data/middleburry/teddy/im2.png\
		test-data/middleburry/teddy/im6.png\
		100 ssd-wta-teddy.png -c

ssd-texture-wta-teddy:
	python -m stereomatch.single_image_app\
		test-data/middleburry/teddy/im2.png\
		test-data/middleburry/teddy/im6.png\
		--cost-method ssd-texture\
		--aggregation-method wta\
		100 ssd-texture-wta-teddy.png -c
