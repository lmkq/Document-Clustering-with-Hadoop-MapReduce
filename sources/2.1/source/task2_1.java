import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FSDataOutputStream;

public class task2_1 {
    private static final int K = 3; // k cluster
    private static final int MAX_ITERATIONS = 20; // max iterations
    private static final double THRESHOLD = 0.00001; // threshold for stop early

    public static class Point {
        private double x, y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double getX() {
            return x;
        }

        public double getY() {
            return y;
        }

        public static Point average(List<Point> points) {
            double sumX = points.stream().mapToDouble(Point::getX).sum();
            double sumY = points.stream().mapToDouble(Point::getY).sum();
            int count = points.size();
            return new Point(sumX / count, sumY / count);
        }

        public double distanceTo(Point other) {
            return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
        }

        public static Point fromString(String str) {
            String[] parts = str.split(",");
            return new Point(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]));
        }

        @Override
        public String toString() {
            String str_point = String.valueOf(x) + "," + String.valueOf(y);
            return str_point;
        }
    }

    // init k centroids randomly
    public static List<Point> initializeCentroids(int k, FileSystem fs, String inputPath) throws IOException {
        List<Point> allPoints = new ArrayList<>();
        List<Point> centroids = new ArrayList<>();

        FileStatus[] files = fs.listStatus(new Path(inputPath));
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split(",");
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        allPoints.add(new Point(x, y));
                    } catch (NumberFormatException e) {
                        continue;
                    }
                }
            }
        }

        // shuffle the list to get random centroids
        Collections.shuffle(allPoints);

        for (int i = 0; i < k; i++) {
            centroids.add(allPoints.get(i));
        }

        return centroids;
    }

    // save the centroids to configuration
    public static void saveCentroidsToConfig(Configuration conf, List<Point> centroids) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < centroids.size(); i++) {
            Point p = centroids.get(i);
            sb.append(p.getX()).append(",").append(p.getY()).append("\n");
        }
        conf.set("centroids", sb.toString());
    }

    // read the centroids from configuration
    public static List<Point> readCentroidsFromConfig(Configuration conf) {
        List<Point> centroids = new ArrayList<>();
        String centroidsStr = conf.get("centroids");
        if (centroidsStr != null && !centroidsStr.isEmpty()) {
            String[] lines = centroidsStr.split("\n");
            for (int i = 0; i < lines.length; i++) {
                String[] parts = lines[i].split(",");
                double x = Double.parseDouble(parts[0]);
                double y = Double.parseDouble(parts[1]);
                centroids.add(i, new Point(x, y));
            }
        }
        return centroids;
    }

    // read centroids from file
    private static List<Point> readCentroidsFromFile(FileSystem fs, Path path) throws IOException {
        List<Point> centroids = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("centroids")) {
                    String[] parts = line.split(",");
                    int clusterIndex = Integer.parseInt(parts[0].substring(parts[0].indexOf(':') + 1));
                    double x = Double.parseDouble(parts[1]);
                    double y = Double.parseDouble(parts[2]);
                    centroids.add(clusterIndex, new Point(x, y));
                }
            }
        }
        return centroids;
    }

    public static class ClusterMapper extends Mapper<Object, Text, IntWritable, Text> {
        private List<Point> centroids;
        private IntWritable clusterKey = new IntWritable();

        @Override
        protected void setup(Context context) {
            centroids = readCentroidsFromConfig(context.getConfiguration());
        }

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts;
            Point point;
            try {
                parts = value.toString().trim().split(",");
                double x, y;
                x = Double.parseDouble(parts[1]);
                y = Double.parseDouble(parts[2]);
                if (parts[0].charAt(0) == 'c') { // skip if it is centroids
                    return;
                }
                point = new Point(x, y);
            } catch (Exception e) {
                return;
            }

            int nearestCluster = getNearestCluster(point);
            clusterKey.set(nearestCluster);

            // key: cluster, value: datapoint
            context.write(clusterKey, new Text(point.toString()));
        }

        private int getNearestCluster(Point point) {
            int nearestCluster = 0;
            double minDistance = Double.MAX_VALUE;

            for (int i = 0; i < centroids.size(); i++) {
                double distance = point.distanceTo(centroids.get(i));
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCluster = i;
                }
            }

            return nearestCluster;
        }
    }

    public static class ClusterReducer extends Reducer<IntWritable, Text, Text, Text> {
        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            List<Point> points = new ArrayList<>();
            for (Text value : values) {
                points.add(Point.fromString(value.toString()));
            }

            Point newCentroid = Point.average(points);

            for (Point point : points) {
                // save the cluster & datapoints: key: cluster, value: datapoint
                context.write(new Text(key.toString() + ","), new Text(point.toString()));

            }

            // save the centroids, key: "centroids:"cluster, value: centroid
            context.write(new Text("centroids:" + key.toString() + ","), new Text(newCentroid.toString()));
        }
    }

    // save the final output into output clusers & output classes
    public static void writeOutput(FileSystem fs, int k, Path inputPath, String outputCluster, String outputClasses)
            throws IOException {
        List<List<Point>> points = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            points.add(new ArrayList<>());
        }

        List<Point> centroids = new ArrayList<>();

        FileStatus[] files = fs.listStatus(inputPath);
        for (FileStatus file : files) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(file.getPath())))) {
                String line;
                while ((line = br.readLine()) != null) {
                    try {
                        String[] parts = line.trim().split(",");
                        int cl;
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        if (parts[0].charAt(0) == 'c') {
                            cl = Integer.parseInt(parts[0].substring(parts[0].indexOf(':') + 1));
                            centroids.add(cl, new Point(x, y));
                        } else {
                            cl = Integer.parseInt(parts[0]);
                            points.get(cl).add(new Point(x, y));
                        }
                    } catch (Exception e) {
                        continue;
                    }
                }
            }
        }

        try (FSDataOutputStream fsOutput = fs.create(new Path(outputClasses));
                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fsOutput))) {
            for (int i = 0; i < k; i++) {
                List<Point> clusterPoints = points.get(i);
                for (Point point : clusterPoints) {
                    writer.write(i + "," + point.getX() + "," + point.getY());
                    writer.newLine();
                }
            }
        }

        try (FSDataOutputStream fsOutput = fs.create(new Path(outputCluster));
                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fsOutput))) {
            for (int i = 0; i < centroids.size(); i++) {
                Point centroid = centroids.get(i);
                writer.write(i + "," + centroid.getX() + "," + centroid.getY());
                writer.newLine();
            }
        }
    }

    public static void runKmeans(Configuration conf, String inputPath, String outputPath) throws Exception {
        FileSystem fs = FileSystem.get(conf);

        // init randomly k centroids
        List<Point> centroids = initializeCentroids(K, fs, inputPath);
        saveCentroidsToConfig(conf, centroids);
        Path centroidOutputPath = new Path(outputPath);

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            Job job = Job.getInstance(conf, "KMeans Iteration " + iteration);
            job.setJarByClass(task2_1.class);

            job.setMapperClass(ClusterMapper.class);
            job.setReducerClass(ClusterReducer.class);

            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(inputPath));
            FileOutputFormat.setOutputPath(job, new Path(outputPath + "/iter_" + iteration));

            job.waitForCompletion(true);

            // update centroids
            centroidOutputPath = new Path(outputPath + "/iter_" + iteration + "/part-r-00000");
            List<Point> newCentroids = readCentroidsFromFile(fs, centroidOutputPath);

            // check if can stop early
            double maxDistance = Double.MIN_VALUE;
            for (int j = 0; j < centroids.size(); j++) {
                double distance = centroids.get(j).distanceTo(newCentroids.get(j));
                if (distance > maxDistance) {
                    maxDistance = distance;
                }
            }

            if (maxDistance < THRESHOLD) {
                break;
            }

            centroids = newCentroids;
            saveCentroidsToConfig(conf, centroids);
        }

        // save the output into task_2_1.cluster & task_2_1.classes
        writeOutput(fs, K, centroidOutputPath, outputPath + "/task_2_1.clusters", outputPath + "/task_2_1.classes");
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        String inputPath = args[0];
        String outputPath = args[1];

        FileSystem fs = FileSystem.get(conf);

        runKmeans(conf, inputPath, outputPath);
    }
}
