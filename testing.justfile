set export 
MAX_DISPARITY := "128"

about:
	@echo Testing shortcuts

ssd-wta-playroom:
	python -m stereomatch.cli_image test-data/middleburry/playroom/im0.png test-data/middleburry/playroom/im1.png 128 ssd-wta-playroom.pgm

ssd-wta-teddy:
	python -m stereomatch.cli_image tests/data/middleburry/teddy/im2.png \
		tests/data/middleburry/teddy/im6.png \
		${MAX_DISPARITY} ssd-wta-teddy.png -c

ssd-sga-dyn-teddy:
	python -m stereomatch.cli_image tests/data/middleburry/teddy/im2.png \
		tests/data/middleburry/teddy/im6.png \
		-cm ssd \
		-am sgm \
		-dm dyn \
		-fig \
		${MAX_DISPARITY} ssd-sga-dyn-teddy.png -c

ssd-texture-wta-teddy:
	python -m stereomatch.cli_image tests/data/middleburry/teddy/im2.png \
		tests/data/middleburry/teddy/im6.png \
		--cost-method ssd-texture \
		--disparity-method-method wta \
		${MAX_DISPARITY} ssd-texture-wta-teddy.png -c


birchfield-wta-teddy:
	python -m stereomatch.cli_image \
		tests/data/middleburry/teddy/im2.png \
		tests/data/middleburry/teddy/im6.png \
		--cost-method birchfield \
		--disparity-method wta \
		${MAX_DISPARITY} birchfield-wta-teddy.png -c
