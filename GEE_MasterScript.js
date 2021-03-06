var table = ee.FeatureCollection("users/milesvandenburg/LaughingBrook_Site"),
    Deciduous = /* color: #98ff00 */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-72.4078137492669, 42.071135246051334]),
            {
              "Landcover": 0,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40778156275871, 42.07144584752443]),
            {
              "Landcover": 0,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.4078137492669, 42.071740519311874]),
            {
              "Landcover": 0,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40880080218437, 42.072146686290104]),
            {
              "Landcover": 0,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.41205163951042, 42.0717126450122]),
            {
              "Landcover": 0,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.41202481742027, 42.07148965017399]),
            {
              "Landcover": 0,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.38594491662036, 42.06607226891508]),
            {
              "Landcover": 0,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.38628823937427, 42.064989047548565]),
            {
              "Landcover": 0,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.43040521325122, 42.05001321156282]),
            {
              "Landcover": 0,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.4158139962102, 42.06951296713091]),
            {
              "Landcover": 0,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.42817361535083, 42.065626238826006]),
            {
              "Landcover": 0,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.42225865341771, 42.046717200065494]),
            {
              "Landcover": 0,
              "system:index": "11"
            })]),
    Conciferous = /* color: #d63000 */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-72.39991885376526, 42.073058893217066]),
            {
              "Landcover": 1,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39957553101135, 42.0726129122166]),
            {
              "Landcover": 1,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.37288218689514, 42.06445727851603]),
            {
              "Landcover": 1,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.36919146729065, 42.06311914480963]),
            {
              "Landcover": 1,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.4012063140924, 42.08153193681643]),
            {
              "Landcover": 1,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40592700195862, 42.07401455623721]),
            {
              "Landcover": 1,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39159327698303, 42.075989547546705]),
            {
              "Landcover": 1,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.384383499151, 42.07318631578446]),
            {
              "Landcover": 1,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.3994038696344, 42.06490331681765]),
            {
              "Landcover": 1,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39910346222473, 42.062035873057816]),
            {
              "Landcover": 1,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39910346222473, 42.06174912155932]),
            {
              "Landcover": 1,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.3991892929132, 42.061334922664145]),
            {
              "Landcover": 1,
              "system:index": "11"
            })]),
    Pasture = /* color: #0b4a8b */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-72.41420413507797, 42.07744231086623]),
            {
              "Landcover": 2,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.4136891509471, 42.076550407334885]),
            {
              "Landcover": 2,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.42816235079147, 42.0677899743949]),
            {
              "Landcover": 2,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.43500734819747, 42.06552799350849]),
            {
              "Landcover": 2,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.43552233232833, 42.06626075670549]),
            {
              "Landcover": 2,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40983258710077, 42.06600546394304]),
            {
              "Landcover": 2,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40110359288667, 42.064768731060724]),
            {
              "Landcover": 2,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40098021127199, 42.06456960678404]),
            {
              "Landcover": 2,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39651115476624, 42.064645298050316]),
            {
              "Landcover": 2,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39644678174989, 42.06517098416207]),
            {
              "Landcover": 2,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39322770234418, 42.08654793426965]),
            {
              "Landcover": 2,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.3932491600163, 42.08607019892846]),
            {
              "Landcover": 2,
              "system:index": "11"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40942824479413, 42.08556857294834]),
            {
              "Landcover": 2,
              "system:index": "12"
            })]),
    Urban_residential = /* color: #ffc82d */ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Point([-72.40088949594553, 42.06363980180735]),
            {
              "Landcover": 3,
              "system:index": "0"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40430126581248, 42.06422125145982]),
            {
              "Landcover": 3,
              "system:index": "1"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39445219430979, 42.065885920333386]),
            {
              "Landcover": 3,
              "system:index": "2"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40005264673289, 42.065911805987966]),
            {
              "Landcover": 3,
              "system:index": "3"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40082780513819, 42.06560515986329]),
            {
              "Landcover": 3,
              "system:index": "4"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39876825596866, 42.06590571478223]),
            {
              "Landcover": 3,
              "system:index": "5"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.41232906209984, 42.066775864923805]),
            {
              "Landcover": 3,
              "system:index": "6"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.41288696157494, 42.06663449122125]),
            {
              "Landcover": 3,
              "system:index": "7"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.3876025871167, 42.07355978272062]),
            {
              "Landcover": 3,
              "system:index": "8"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39740042404742, 42.06810059170047]),
            {
              "Landcover": 3,
              "system:index": "9"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.40496425346942, 42.064436797319246]),
            {
              "Landcover": 3,
              "system:index": "10"
            }),
        ee.Feature(
            ee.Geometry.Point([-72.39554155890517, 42.0859459643795]),
            {
              "Landcover": 3,
              "system:index": "11"
            })]);

///BRING IN THE ROSS FOR THE CLOUDS
//ROSSAW IS READY



// //Importing imagery from Landsat 7 and creating TCC and FCCs from the data
// var dataset = ee.ImageCollection('LANDSAT/LE07/C01/T1_RT')
//                   .filterDate('1999-01-01', '2002-12-31');
// var trueColor321 = dataset.select(['B3', 'B2', 'B1']);
// var trueColor321Vis = {};

// var falseColor432 = dataset.select(['B4', 'B3', 'B2']);
// var falseColor432FCC = {}

// Map.setCenter(6.746, 46.529, 6);
// Map.addLayer(falseColor432, falseColor432FCC, 'False Color (432)');
// Map.addLayer(trueColor321, trueColor321Vis, 'True Color (321)');


var laughingbrook = ee.FeatureCollection(table);
laughingbrook = laughingbrook.geometry();
var sitebuffer = laughingbrook.buffer(3000);

Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(laughingbrook, {}, 'default display');
Map.addLayer(laughingbrook, {color: 'FF0000'}, 'colored', false);


var cloud_filtered = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};





//Brought in Landsat 7 imagery corrected for atmosphere 'SR'
var dataset = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterDate('2002-06-01', '2002-07-30').map(cloud_filtered);
var visParams = {
  bands: ['B3', 'B2', 'B1'],
  min: 0,
  max: 3000,
  gamma: 1.4,
};


// Get the least cloudy image in Image.
var cloud_filtere = ee.Image(
  dataset.filterBounds(laughingbrook)
    .filterDate('2002-06-01', '2002-07-30')
    .sort('CLOUD_COVER')
    .first()
);

print(cloud_filtere, 'Filtered Image')



var dataset_med = dataset.median();
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(dataset_med, visParams, 'TCC_Summer', false);

print(dataset_med)

//Define Merge for Classes
var newfc = Urban_residential.merge(Pasture).merge(Conciferous).merge(Deciduous);
print(newfc,'newfc')


// Use these bands for classification
var bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'];
// The name of the property on the points storing the class label


// Sample the composite to generate training data.  Note that the
// class label is stored in the 'landcover' property
var training = dataset_med.select(bands).sampleRegions({
  collection: newfc,
  properties: ['Landcover'],
  scale: 30
});



print(training, "training")


// Train a Random Forest classifier.
var classifier = ee.Classifier.randomForest().train({
  features: training,
  classProperty: 'Landcover',
  inputProperties: bands
});


// // Classify the input imagery.

var classified = dataset_med.select(bands).classify(classifier);
var classified_clip = classified.clip(sitebuffer);

print(classified, "classified")

 //Define a palette for the Land Use classification.
var palette = [
  '00ff00', // Deciduous (0)  // Light Green
  '008000', // Conciferous (1) // Dark Green
  'yellow', // Pasture (2)  // Orange
  'blue', // Urban_residential (3)  // TEAL
];


Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clip, {min: 0, max: 3, palette: palette}, 'Land Use Classification');





var options3 = {
  title: 'Landsat 7 Tier 1 SR spectra Clsasifcation Distribution',
  hAxis: {title: 'Bands'},
  vAxis: {title: 'Reflectance'},
  lineWidth: 1,
  pointSize: 4,
  series: {
    0: {color: '00FF00'}, // Deciduous
    1: {color: '0000FF'}, // Conciferous
    2: {color: 'FF0000'}, // Pasture
    3: {color: 'orange'} // Urban Residential
}};

// Define a list of Landsat 8 wavelengths for X-axis labels.
var wavelengths = [0.44, 0.48, 0.56, 0.65, 0.86, 1.61, 2.35];



var options3 = {
  title: 'Landsat 7 Tier 1 SR spectra',
  hAxis: {title: 'Bands (micrometers)'},
  vAxis: {title: 'Reflectance'},
  lineWidth: 1,
  pointSize: 4,
  series: {
    0: {color: '00FF00'}, // Deciduous
    1: {color: '0000FF'}, // Conciferous
    2: {color: 'FF0000'}, // Pasture
    3: {color: 'orange'} // Urban Residential
}};

// Define a list of Landsat 8 wavelengths for X-axis labels.
var wavelengths = [0.44, 0.48, 0.56, 0.65, 0.86, 1.61, 2.35];

// Create the chart and set options.
var spectraChart = ui.Chart.image.regions(
    dataset_med.select(bands), newfc, ee.Reducer.mean(), 30, 'Landcover', bands)
        .setChartType('ScatterChart')
        .setOptions(options3);

// Display the chart.
print(spectraChart);


var dataset_reducedbands = dataset_med.select(bands);

print(dataset_reducedbands, "names")

// Reduce the region for Pasture. The region parameter is the Feature geometry of PASTURE.
var meanDictionaryPasture = dataset_reducedbands.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: Pasture,
  scale: 30,
  maxPixels: 1e9
});

print(meanDictionaryPasture, "Mean Dictionary")


var PastureMeanValues = ee.Feature(meanDictionaryPasture.select(dataset_reducedbands.bandNames()))



// Print the first feature, to illustrate the result.
print(ee.Feature(meanDictionaryPasture.select(dataset_reducedbands.bandNames())), "Pasture Mean Values");



// Reduce the region for Urban. The region parameter is the Feature geometry of Urban_residential.
var meanDictionaryUrban = dataset_reducedbands.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: Urban_residential,
  scale: 30,
  maxPixels: 1e9
});


var UrbanMeanValues = ee.Feature(meanDictionaryUrban.select(dataset_reducedbands.bandNames()))



// Print the first feature, to illustrate the result.
print(ee.Feature(meanDictionaryUrban.select(dataset_reducedbands.bandNames())), "Urban Mean Values");



// Reduce the region for Conciferous Trees. The region parameter is the Feature geometry of Conciferous.
var meanDictionaryConciferous = dataset_reducedbands.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: Conciferous,
  scale: 30,
  maxPixels: 1e9
});


var ConciferousMeanValues = ee.Feature(meanDictionaryConciferous.select(dataset_reducedbands.bandNames()))



// Print the Conciferous feature, to illustrate the result.
print(ee.Feature(meanDictionaryConciferous.select(dataset_reducedbands.bandNames())), "Conciferous Mean Values");






// Reduce the region for Deciduous Trees. The region parameter is the Feature geometry of Deciduous.
var meanDictionaryDeciduous = dataset_reducedbands.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: Deciduous,
  scale: 30,
  maxPixels: 1e9
});


var DeciduousMeanValues = ee.Feature(meanDictionaryDeciduous.select(dataset_reducedbands.bandNames()));



// Print the Deciduous feature, to illustrate the result.
print(ee.Feature(meanDictionaryDeciduous.select(dataset_reducedbands.bandNames())), "Deciduous Mean Values");







///Additional Color Composites




var dataset_winter = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterDate('2002-01-01', '2002-03-15').map(cloud_filtered);
var visParams = {
  bands: ['B3', 'B2', 'B1'],
  min: 0,
  max: 3000,
  gamma: 1.4,
};

var dataset_med_winter = dataset_winter.median();


var centroid = laughingbrook.centroid();
Map.addLayer(centroid, {}, 'centroid');

// // Get the least cloudy image in Image.
// var cloud_filtered_image = ee.Image(
//   dataset.filterBounds(centroid)
//     .filterDate('2002-04-01', '2002-04-30')
//     .sort('CLOUD_COVER')
//     .first()
// );




// Create a circle by drawing a 20000 meter buffer around a point.
//var roi = geometry.buffer(8000);  

//Compute a 3000m buffer of the polygon.
var sitebuffer = laughingbrook.buffer(3000);



// Display a clipped version of the TCC.
var clipped = dataset_med.clip(sitebuffer);
Map.addLayer(clipped, visParams, 'Clipped Image_TCC', false);
print(clipped)


// Display a clipped version of the Winter TCC.
var clipped_winter = dataset_med_winter.clip(sitebuffer);
Map.addLayer(clipped_winter, visParams, 'Clipped Image_TCC_Winter', false);
print(clipped_winter)





//Map.addLayer(cloud_filtered_image.clip(sitebuffer));


//Created a false color composite of the Summer Image 
var fcc432 = dataset_med.select(['B4', 'B3', 'B2']);
var fcc432vizparams = { 
    min: 0,
    max: 3000,
    gamma: 1.4,};


// Display a clipped version of the FCC.
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(fcc432.clip(sitebuffer), fcc432vizparams, 'False Color (432)_Summer', false);


//Created a false color composite of the Winter Image 
var fcc432_Winter = dataset_med_winter.select(['B4', 'B3', 'B2']);
var fcc432vizparams_winter = { 
    min: 0,
    max: 3000,
    gamma: 1.4};



// Display a clipped version of the FCC.
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(fcc432_Winter.clip(sitebuffer), fcc432vizparams_winter, 'False Color (432)_Winter', false);







// Map.centerObject(image, 9);
// var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
// Map.addLayer(ndvi, ndviParams, 'NDVI image');



//Imported the data into GEE and added it to the map coloring it in red. 

var laughingbrook = ee.FeatureCollection(table);
laughingbrook = laughingbrook.geometry();


Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(laughingbrook, {}, 'default display');
Map.addLayer(laughingbrook, {color: 'FF0000'}, 'colored');



// Compute the Normalized Difference Vegetation Index (NDVI).
// var nir = dataset_med.select('B4');
// var red = dataset_med.select('B3');
// var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

var winter_ndvi = clipped_winter.normalizedDifference(['B4', 'B3']);


// Display the result.
Map.centerObject(laughingbrook, 12);
var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
Map.addLayer(winter_ndvi, ndviParams, 'Winter_NDVI image', false);



// Compute the Normalized Difference Vegetation Index (NDVI) for Winter Months.
// var nir = dataset_med.select('B4');
// var red = dataset_med.select('B3');
// var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');

var ndvi = clipped.normalizedDifference(['B4', 'B3']);


// Display the result.
Map.centerObject(laughingbrook, 12);
var ndviParams = {min: 0, max: 1, palette: ['red', 'Yellow', 'Green']};
Map.addLayer(ndvi, ndviParams, 'NDVI image', false);



// Compute the Normalized Difference Water Index (NDWI).
var ndwi = clipped.normalizedDifference(['B4', 'B5']);

// Display the result.
Map.centerObject(laughingbrook, 12);
var ndwiParams = {min: 0, max: .8, palette: ['0000ff', '00ffff', 'ffff00', 'ff0000', 'ffffff']};
Map.addLayer(ndwi, ndwiParams, 'NDWI image', false);




// Compute the Normalized Difference Built-In Index (NDBI).
var ndbi = clipped.normalizedDifference(['B5', 'B4']);

// Display the result.
Map.centerObject(laughingbrook, 12);
var ndbiParams = {min: -1, max: 0, palette: ['0000ff', '00ffff', 'ffff00', 'ff0000', 'ffffff']};
Map.addLayer(ndbi, ndbiParams, 'NDBI image', false);

print(ndbi, "ndbistuffs")




// Compute the EVI using an expression.
var evi = clipped.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': clipped.select('B4'),
      'RED': clipped.select('B3'),
      'BLUE': clipped.select('B1')
});

//Define 
var eviParams = {min: 0, max: 1, palette: ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901',
    '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01',
    '012E01', '011D01', '011301']};


// Display the result.
Map.centerObject(laughingbrook, 12);
Map.addLayer(evi, eviParams, 'EVI', false);
    
// 500m, 1000m, 1500m, 2000m, 2500m, 3000m buffers 


//Clip Site to classified image
var classified_clipSite = classified.clip(laughingbrook);
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clipSite, {min: 0, max: 3, palette: palette}, 'Land Use Classification For LaughingBrook');



// Compute a 500m buffer of the polygon.
var buffer500 = laughingbrook.buffer(500);
//Clip Buffer to classified image
var classified_clip500 = classified.clip(buffer500);
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clip500, {min: 0, max: 3, palette: palette}, 'Land Use Classification with 500 Buffer');

//Compute a 1000m buffer of the polygon.
var buffer1000 = laughingbrook.buffer(1000);
//Clip Buffer to classified image
var classified_clip1000 = classified.clip(buffer1000);
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clip1000, {min: 0, max: 3, palette: palette}, 'Land Use Classification with 1000 Buffer');


//Compute a 1500m buffer of the polygon.
var buffer1500 = laughingbrook.buffer(1500);
//Clip Buffer to classified image
var classified_clip1500 = classified.clip(buffer1500);
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clip1500, {min: 0, max: 3, palette: palette}, 'Land Use Classification with 1500 Buffer');


//Compute a 2000m buffer of the polygon.
var buffer2000 = laughingbrook.buffer(2000);
//Clip Buffer to classified image
var classified_clip2000 = classified.clip(buffer2000);
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clip2000, {min: 0, max: 3, palette: palette}, 'Land Use Classification with 2000 Buffer');


//Compute a 2500m buffer of the polygon.
var buffer2500 = laughingbrook.buffer(2500);
//Clip Buffer to classified image
var classified_clip2500 = classified.clip(buffer2500);
Map.setCenter(-72.404487, 42.064625, 13);
Map.addLayer(classified_clip2500, {min: 0, max: 3, palette: palette}, 'Land Use Classification with 2500 Buffer');


//Compute a 3000m buffer of the polygon.
var buffer3000 = laughingbrook.buffer(3000);



// Compute the centroid of the polygon.
//var centroid = laughingbrook.centroid();
//Map.addLayer(centroid, {}, 'centroid');

//Add Buffer layers 
Map.addLayer(buffer500, {}, 'buffer500', false);
Map.addLayer(buffer1000, {color: 'red'}, 'buffer1000', false);
Map.addLayer(buffer1500, {color: 'yellow'}, 'buffer1500', false);
Map.addLayer(buffer2000, {color: 'blue'}, 'buffer2000', false);
Map.addLayer(buffer2500, {color: 'orange'}, 'buffer2500', false);
Map.addLayer(buffer3000, {}, 'buffer3000', false);



var centroid = laughingbrook.centroid();
Map.addLayer(centroid, {}, 'centroid');




// Create a geometry representing an export region.

// // Export the image, specifying scale and region.
Export.image.toDrive({
  image: dataset_med,
  description: 'Dataset_Med_NonTiff',
  scale: 30,
  region:dataset_med
});



// ACCURACY ASSESSMENT:
var withRandom = training.randomColumn();
print(withRandom);

// Approximately 70% of our training data
var trainingPartition = withRandom.filter(ee.Filter.lt('random', 0.7));
// Approximately 30% of our training data
var testingPartition = withRandom.filter(ee.Filter.gte('random', 0.7));

// Trained with 70% of our data.
var trainedClassifier = ee.Classifier.randomForest().train({
  features: trainingPartition, 
  classProperty: 'Landcover', 
  inputProperties: bands
});

var test = testingPartition.classify(trainedClassifier);
print(test, 'testy');

//Creates error matrix using the actual and the predicted values 

//The Confusion Matrix represents expected accuracy.

var errorMatrix = test.errorMatrix('Landcover', 'classification');

var OA = errorMatrix.accuracy()
var CA = errorMatrix.consumersAccuracy()
var Kappa = errorMatrix.kappa()
var Order = errorMatrix.order()
var PA = errorMatrix.producersAccuracy()
 
print(errorMatrix,'Error Matrix')
print(OA,'Overall Accuracy Error Matrix')
print(CA,'Consumers Accuracy Error Matrix')
print(Kappa,'Kappa Error Matrix')
print(Order,'Order Error Matrix')
print(PA,'Producers Accuracy Error Matrix')


//Create Confusion Matrix for actual Data 
var confMatrix = classifier.confusionMatrix();


var OA = confMatrix.accuracy()
var CA = confMatrix.consumersAccuracy()
var Kappa = confMatrix.kappa()
var Order = confMatrix.order()
var PA = confMatrix.producersAccuracy()
 
print(confMatrix,'Confusion Matrix')
print(OA,'Overall Accuracy')
print(CA,'Consumers Accuracy')
print(Kappa,'Kappa')
print(Order,'Order')
print(PA,'Producers Accuracy')





// var exportNewfc = ee.Feature(null, {matrix: newfc.array()})

// // Export the FeatureCollection.
// Export.table.toDrive({
//   collection: ee.FeatureCollection(newfc),
//   description: 'exportAccuracy',
//   fileFormat: 'CSV'
// });




// CHART RESULTS:
var options = {
  lineWidth: 1,
  pointSize: 2,
  hAxis: {title: 'Classes'},
  vAxis: {title: 'Area m^2'},
  title: 'Area by class',
  series: {
    0: { color:'00ff00'},
    1: { color: '008000'},
    2: { color: 'yellow'},
    3: { color: 'blue'}
  }
};




// '00ff00', // Deciduous (0)  // Light Green
// '008000', // Conciferous (1) // Dark Green
//  'yellow', // Pasture (2)  // Orange
//  'blue', // Urban_residential (3)  // TEAL


// Pre-define some customization options.
var options1 = {
  title: 'Landsat 7 DN histogram, bands 1-7',
  fontSize: 20,
  hAxis: {title: 'DN'},
  vAxis: {title: 'count of DN'},
  series: {
    0: {color: 'blue'},
    1: {color: 'green'},
    2: {color: 'red'},
    3: {color: 'magenta'},
    4: {color: 'orange'},
    5: {color: 'purple'},
    6: {color: 'yellow'}
  }};

// Make the histogram, set the options.
var histogram = ui.Chart.image.histogram(clipped, sitebuffer, 20, 20)
    .setSeriesNames(['blue', 'green', 'red', 'NIR', 'SWIR', "SWIR2", "Thermal"])
    .setOptions(options1);

// Display the histogram.
print(histogram);



// // Reduce the region. The region parameter is the Feature geometry.
// var meanDictionary = clipped.reduceRegion({
//   reducer: ee.Reducer.mean(),
//   geometry: region.geometry(),
//   scale: 30,
//   maxPixels: 1e9
// });

// // The result is a Dictionary.  Print it.
// print(meanDictionary);


print("Total Meters Squared of site", ee.Image.pixelArea().addBands(classified_clip));


//Area Chart 3000m
var areaChart = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: buffer3000,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);

print('3000m Buffer area: ',areaChart)

// Print polygon area in square kilometers.
print('3000m Buffer area in meters squared: ', sitebuffer.area());


//Area Chart 2500m
var areaChart_2500 = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: buffer2500,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);



print('2500m Buffer area: ', areaChart_2500);

// Print polygon area in square kilometers.
print('2500m Buffer area in meters squared: ', buffer2500.area());

//Area Chart 2000m
var areaChart_2000 = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: buffer2000,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);


print('2000m Buffer area: ', areaChart_2000);

// Print polygon area in square kilometers.
print('2000m Buffer area in meters squared:: ', buffer2000.area());



//Area Chart 1500m
var areaChart_1500 = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: buffer1500,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);



print('1500m Buffer area in meters squared:: ', areaChart_1500);

// Print polygon area.
print('1500m Buffer area: ', buffer1500.area());

//Area Chart 1000m
var areaChart_1000 = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: buffer1000,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);

print('1000m Buffer area: ',areaChart_1000);

// Print polygon area in square kilometers.
print('1000m Buffer area in meters squared: ', buffer1000.area());


//Area Chart 500m
var areaChart_500 = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: buffer500,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);

print('500m Buffer area: ', areaChart_500);

// Print polygon area in square kilometers.
print('500m Buffer area meters Squared: ', buffer500.area());

//Area Chart Laughing Brook Site
var areaChart_500 = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified_clip),
  classBand: 'classification', 
  region: laughingbrook,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setSeriesNames(['Deciduous', 'Conciferous', 'Pasture', 'Urban_residential']).setOptions(options);

print('Laughing Brook Site area: ', areaChart_500);

// Print polygon area in square meters.
print('Laughing Brook Site area meters Squared: ', laughingbrook.area());


//MEAN VALUE CALCULATIONS FOR INDICES

//NDVI




// Reduce the region for the Laughing Brook Site. The region parameter is the Feature geometry of the site.
var ndviMean_site = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: laughingbrook,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Laughing Brook Site NDVI Mean: ',ndviMean_site);

// Reduce the region for the Buffer 3000 site. The region parameter is the Feature geometry of the site.
var ndviMean_buffer3000 = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer3000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 3000 NDVI Mean: ', ndviMean_buffer3000);




// Reduce the region for the Buffer 2500 site. The region parameter is the Feature geometry of the site.
var ndviMean_buffer2500 = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2500 NDVI Mean: ', ndviMean_buffer2500);


// Reduce the region for the Buffer 2000 site. The region parameter is the Feature geometry of the site.
var ndviMean_buffer2000 = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2000 NDVI Mean: ', ndviMean_buffer2000);


// Reduce the region for the Buffer 1500 site. The region parameter is the Feature geometry of the site.
var ndviMean_buffer1500 = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1500 NDVI Mean: ', ndviMean_buffer1500);

// Reduce the region for the Buffer 1000 site. The region parameter is the Feature geometry of the site.
var ndviMean_buffer1000 = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1000 NDVI Mean: ', ndviMean_buffer1000);




// Reduce the region for the Buffer 500 site. The region parameter is the Feature geometry of the site.
var ndviMean_buffer500 = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 500 NDVI Mean: ', ndviMean_buffer500);





//NDWI

// Make the histogram, set the options.
var histogramndwi = ui.Chart.image.histogram(ndwi, sitebuffer, 30, 20)
  

print(histogramndwi, "NDWI histogram");



// Reduce the region for the Laughing Brook Site. The region parameter is the Feature geometry of the site.
var ndwiMean_site = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: laughingbrook,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Laughing Brook Site NDWI Mean: ', ndwiMean_site);


// Reduce the region for the Buffer 3000 site. The region parameter is the Feature geometry of the site.
var ndwiMean_buffer3000 = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer3000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 3000 NDWI Mean: ', ndwiMean_buffer3000);




// Reduce the region for the Buffer 2500 site. The region parameter is the Feature geometry of the site.
var ndwiMean_buffer2500 = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2500 NWVI Mean: ', ndwiMean_buffer2500);


// Reduce the region for the Buffer 2000 site. The region parameter is the Feature geometry of the site.
var ndwiMean_buffer2000 = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2000 NDWI Mean: ', ndwiMean_buffer2000);


// Reduce the region for the Buffer 1500 site. The region parameter is the Feature geometry of the site.
var ndwiMean_buffer1500 = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1500 NDWI Mean: ', ndwiMean_buffer1500);

// Reduce the region for the Buffer 1000 site. The region parameter is the Feature geometry of the site.
var ndwiMean_buffer1000 = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1000 NDWI Mean: ', ndwiMean_buffer1000);




// Reduce the region for the Buffer 500 site. The region parameter is the Feature geometry of the site.
var ndwiMean_buffer500 = ndwi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 500 NDWI Mean: ', ndwiMean_buffer500);





//NDBI

// // Pre-define some customization options.
// var options1 = {
//   title: 'Landsat 7 DN histogram, bands NDVI',
//   fontSize: 20,
//   hAxis: {title: 'DN'},
//   vAxis: {title: 'count of DN'},
//   series: {
//     0: {color: 'blue'},
//     1: {color: 'green'},
//     2: {color: 'red'},
//     3: {color: 'magenta'},
//     4: {color: 'orange'},
//     5: {color: 'purple'},
//     6: {color: 'yellow'}
//   }};

// Make the histogram, set the options.
var histogramndbi = ui.Chart.image.histogram(ndbi, sitebuffer, 30, 20)
  

print(histogramndbi, "NDBI histogram");



// Reduce the region for the Laughing Brook Site. The region parameter is the Feature geometry of the site.
var ndbiMean_site = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: laughingbrook,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Laughing Brook Site NDBI Mean: ', ndbiMean_site);


// Reduce the region for the Buffer 3000 site. The region parameter is the Feature geometry of the site.
var ndbiMean_buffer3000 = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer3000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 3000 NDBI Mean: ', ndbiMean_buffer3000);




// Reduce the region for the Buffer 2500 site. The region parameter is the Feature geometry of the site.
var ndbiMean_buffer2500 = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2500 NDBI Mean: ', ndbiMean_buffer2500);


// Reduce the region for the Buffer 2000 site. The region parameter is the Feature geometry of the site.
var ndbiMean_buffer2000 = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2000 NDBI Mean: ', ndbiMean_buffer2000);


// Reduce the region for the Buffer 1500 site. The region parameter is the Feature geometry of the site.
var ndbiMean_buffer1500 = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1500 NDBI Mean: ', ndbiMean_buffer1500);

// Reduce the region for the Buffer 1000 site. The region parameter is the Feature geometry of the site.
var ndbiMean_buffer1000 = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1000 NDBI Mean: ', ndbiMean_buffer1000);




// Reduce the region for the Buffer 500 site. The region parameter is the Feature geometry of the site.
var ndbiMean_buffer500 = ndbi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 500 NDBI Mean: ', ndbiMean_buffer500);



//EVI

// Reduce the region for the Laughing Brook Site. The region parameter is the Feature geometry of the site.
var eviMean_site = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: laughingbrook,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Laughing Brook Site EVI Mean: ', eviMean_site);


// Reduce the region for the Buffer 3000 site. The region parameter is the Feature geometry of the site.
var eviMean_buffer3000 = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer3000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 3000 EVI Mean: ', eviMean_buffer3000);




// Reduce the region for the Buffer 2500 site. The region parameter is the Feature geometry of the site.
var eviMean_buffer2500 = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2500 EVI Mean: ', eviMean_buffer2500);


// Reduce the region for the Buffer 2000 site. The region parameter is the Feature geometry of the site.
var eviMean_buffer2000 = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer2000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 2000 EVI Mean: ', eviMean_buffer2000);


// Reduce the region for the Buffer 1500 site. The region parameter is the Feature geometry of the site.
var eviMean_buffer1500 = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1500 EVI Mean: ', eviMean_buffer1500);

// Reduce the region for the Buffer 1000 site. The region parameter is the Feature geometry of the site.
var eviMean_buffer1000 = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer1000,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 1000 EVI Mean: ', eviMean_buffer1000);




// Reduce the region for the Buffer 500 site. The region parameter is the Feature geometry of the site.
var eviMean_buffer500 = evi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: buffer500,
  scale: 30,
  maxPixels: 1e9
});

// The result is a Dictionary.  Print it.
print('Buffer 500 EVI Mean: ', eviMean_buffer500);

Map.addLayer(laughingbrook, {color: 'FF0000'}, 'other');



//Projection of the site
var b2proj = dataset_med.select('B2').projection();
print('Band 1 projection: ', b2proj); // ee.Projection object

//Date for the site
var date = ee.Date(dataset_med.get('system:time_start'));

print('Timestamp: ', date); // ee.Date

print(['LANDSAT_PRODUCT_ID']);


// Get a list of all metadata properties.
var properties = dataset_med.propertyNames();
print('Metadata properties: ', properties); // ee.List of metadata properties

print(dataset, "dataset")


// Export the FeatureCollection to a SHP file.
Export.table.toDrive({
  collection: newfc,
  description:'vectorsToDriveExample',
  fileFormat: 'SHP'
});

