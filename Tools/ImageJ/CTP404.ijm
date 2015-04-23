
runMacro("CTP404_FBP_Analysis");
//runMacro("CTP404_Analysis","FBP_image_h");

basename = "x_";
iterations = 12;				// # of iterations to analyze from image reconstruction


// Repeat for each iteration
for (iteration=0; iteration <= iterations; iteration++)
	runMacro("CTP404_Analysis", d2s(iteration,0));

