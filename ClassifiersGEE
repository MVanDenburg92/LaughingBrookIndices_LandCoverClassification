var geometry = /* color: #d63000 */ee.Geometry.Polygon(
        [[[-122.53304372558597, 37.89710508140966],
          [-122.4575127197266, 37.84181641588064],
          [-122.47124562988284, 37.94043970438186]]]);

// Function to cloud mask from the pixel_qa band of Landsat 8 SR data.
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = 1 << 3;
  var cloudsBitMask = 1 << 5;

  // Get the pixel QA band.
  var qa = image.select('pixel_qa');

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      .and(qa.bitwiseAnd(cloudsBitMask).eq(0));

  // Return the masked image, scaled to TOA reflectance, without the QA bands.
  return image.updateMask(mask).divide(10000)
      .select("B[0-9]*")
      .copyProperties(image, ["system:time_start"]);
}

// Map the function over one year of data.
var collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
    .filterDate('2016-01-01', '2016-12-31')
    .map(maskL8sr)

var composite = collection.median();

// Display the results.
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3});

// Get some pre-made demonstration labels.
var labels = ee.FeatureCollection('users/nclinton/demo_landcover_labels');
Map.addLayer(labels, {}, 'labels');
print(labels.first()); // Just to check.

var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'];

var training = composite.select(bands).sampleRegions({
  collection: labels, 
  properties: ['landcover'], 
  scale: 30
});
// print(training)

var classifier = ee.Classifier.cart().train({
  features: training, 
  classProperty: 'landcover', 
  inputProperties: bands
});
print(classifier.explain());

var classified = composite.select(bands).classify(classifier);
Map.addLayer(classified, {min: 0, max: 2, palette: ['red', 'green', 'blue']}, 'cart');


// ACCURACY ASSESSMENT:
var withRandom = training.randomColumn();
// print(withRandom);

// Approximately 70% of our training data
var trainingPartition = withRandom.filter(ee.Filter.lt('random', 0.7));
// Approximately 30% of our training data
var testingPartition = withRandom.filter(ee.Filter.gte('random', 0.7));

// Trained with 70% of our data.
var trainedClassifier = ee.Classifier.cart().train({
  features: trainingPartition, 
  classProperty: 'landcover', 
  inputProperties: bands
});

var test = testingPartition.classify(trainedClassifier);
// print(test);

var confusionMatrix = test.errorMatrix('landcover', 'classification');
print(confusionMatrix);


// CHART RESULTS:
var options = {
  lineWidth: 1,
  pointSize: 2,
  hAxis: {title: 'Classes'},
  vAxis: {title: 'Area m^2'},
  title: 'Area by class',
  series: {
    0: { color: 'red'},
    1: { color: 'green'},
    2: { color: 'blue'}
  }
};

var areaChart = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classified),
  classBand: 'classification', 
  region: geometry,
  scale: 30,
  reducer: ee.Reducer.sum()
}).setOptions(options)
  .setSeriesNames(['urban', 'vegetation', 'water']);
print(areaChart);


// CLASSIFIERS:
var classifier = ee.Classifier.randomForest(10).train({
  features: training, 
  classProperty: 'landcover', 
  inputProperties: bands,
});

var classifier2 = ee.Classifier.naiveBayes().train({
  features: training, 
  classProperty: 'landcover', 
  inputProperties: bands,
});

var classifier3 = ee.Classifier.gmoMaxEnt().train({
  features: training, 
  classProperty: 'landcover', 
  inputProperties: bands,
});

var classified1 = composite.classify(classifier);
var classified2 = composite.classify(classifier2);
var classified3 = composite.classify(classifier3);

var mode = classified1.addBands(classified2).addBands(classified3)
    .reduce(ee.Reducer.mode());
Map.addLayer(mode, {min: 0, max: 2, palette: ['red', 'green', 'blue']}, 'mode');

Map.addLayer(classified1, {min: 0, max: 2, palette: ['red', 'green', 'blue']}, 'Random Forest');

