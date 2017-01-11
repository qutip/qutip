#!/bin/bash
htmlblddir="./_build/html"

echo "moving _images to images"
mv $htmlblddir"/_images" $htmlblddir"/images"
echo "fixing images links"
find $htmlblddir -type f -name "*.html" -exec sed -i 's/_images/images/g' {} +

echo "moving _static to static"
mv $htmlblddir"/_static" $htmlblddir"/static"
echo "fixing static links"
find $htmlblddir -type f -name "*.html" -exec sed -i 's/_static/static/g' {} +

echo "moving _modules to modules"
mv $htmlblddir"/_modules" $htmlblddir"/modules"
echo "fixing modules links"
find $htmlblddir -type f -name "*.html" -exec sed -i 's/_modules/modules/g' {} +

echo "moving _sources to sources"
mv $htmlblddir/"_sources" $htmlblddir"/sources"
echo "fixing sources links"
find $htmlblddir -type f -name "*.html" -exec sed -i 's/_sources/sources/g' {} +

