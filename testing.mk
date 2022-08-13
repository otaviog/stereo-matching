about:
	@echo Testing shortcuts

MAX_DISPARITY := 128

ssd-wta-playroom:
	python -m stereomatch.single_image_app test-data/middleburry/playroom/im0.png test-data/middleburry/playroom/im1.png 128 ssd-wta-playroom.pgm

ssd-wta-teddy:
	python -m stereomatch.single_image_app\
		test-data/middleburry/teddy/im2.png\
		test-data/middleburry/teddy/im6.png\
		${MAX_DISPARITY} ssd-wta-teddy.png -c

ssd-texture-wta-teddy:
	python -m stereomatch.single_image_app\
		test-data/middleburry/teddy/im2.png\
		test-data/middleburry/teddy/im6.png\
		--cost-method ssd-texture\
		--aggregation-method wta\
		${MAX_DISPARITY} ssd-texture-wta-teddy.png -c


birchfield-wta-teddy:
	python -m stereomatch.single_image_app\
		test-data/middleburry/teddy/im2.png\
		test-data/middleburry/teddy/im6.png\
		--cost-method birchfield\
		--aggregation-method wta\
		${MAX_DISPARITY} birchfield-wta-teddy.png -c
