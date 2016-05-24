macro "multiplotting"
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//***************************************************************************************************************************************************************************************************//
	//************************* Close all windows and clear results table *******************************************************************************************************************************//
	//***************************************************************************************************************************************************************************************************//
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	close_all_windows		(true, true);
	setBatchMode			(true);	
	//setBatchMode			(false);
	print_section_separator("Generating multi-curve plots vs target test parameter for a user specified parameter, with each plot having individual curves for each value of this parameter, then writing the plot data and resulting images to disk as PNG", true);
	//print_section_separator("Generating multi-curve plots vs target test parameter for ", true);
	//print_section_separator("Generating multi-curve plots vs target test parameter forGenerating multi-curve plots vs target test parameter for", true);
	//print_section_separator("Generating multi-curve plots vs target test parameter for ", false);
	//print_section_separator("Generating multi-curve plots vs target test parameter forGenerating multi-curve plots vs target test parameter for", false);
	//exit();
	//***************************************************************************************************************************************************************************************************//
	//************* Execution control Booleans and general parameter value test and multiplot behavior Booleans ************************************************************************************************************************************//
	//***************************************************************************************************************************************************************************************************//
	simulated_scan 														= true;
	experimental_scan 													= !simulated_scan;
	log_printing														= true;			
	printing_ROI_definitions											= true;		
	printing_reconstructed_image_analysis_info							= true;
	printing_parameter_value_test_info									= true;
	printing_multiplot_parameter_info									= true;	
	printing_input_output_filenames										= true;	
	print_ROI_definitions_path											= false;		
	
	print_missing_paths													= true;
	print_directories_created											= true;		
	print_input_data_path												= true;
	print_input_CSV_path												= false;
	print_input_TXT_path												= false;
	print_input_PNG_paths												= false;
	print_output_CSV_path												= true;
	print_output_TXT_path												= true;
	print_output_PNG_paths												= false;
	print_multiplot_animation_paths 									= false;		
	print_multiplot_GIF_paths 											= true;		
	print_multiplot_AVI_paths 											= true;		
	print_copied_file_paths												= false;
	close_input_PNG_images												= true;
	close_output_PNG_images												= true;
	close_stack_images													= true;		
	close_multiplot_stack_images 										= true;									
	specify_multiplot_parameter_prefix									= true;
	specify_multiplot_parameter_number									= true && !specify_multiplot_parameter_prefix;
	specify_multiplot_parameter_index									= true && !specify_multiplot_parameter_prefix && !specify_multiplot_parameter_number;
	write_folder_strings 												= false;
	write_path_strings 													= false;		
	//***************************************************************************************************************************************************************************************************//
	//************* Parameter value test Booleans ************************************************************************************************************************************//
	//***************************************************************************************************************************************************************************************************//	
	check_all_paths_existence 											= true;
	analyze_specific_data 												= false;
	only_perform_missing_analyses 										= false;
	generate_averaged_data												= false;
	generate_specific_averaged_data										= false;
	generate_averaged_multiplot_data									= false;
	perform_all_missing_analyses										= false;
	perform_multiplot_analyses											= true;
	exit_after_analyzing_specific_data									= true;
	exit_after_performing_all_missing_analyses							= true;
	exit_after_performing_all_parameter_test_analyses					= false;
	exit_after_performing_multiplot_ROI_analyses						= false;
	exit_after_averaging_specific_data									= false;
	
	analyze_compared_data 												= false;
	add_comparison_lines_2_plots										= true;
	add_compared_data													= true;
	add_compared_RSP_data 												= true;
	add_compared_RSP_error_data 										= true;
	add_compared_std_dev_data 											= true;
	add_compared_TV_data 												= true;
	
	//***************************************************************************************************************************************************************************************************//
	//************* Parameter value test Booleans ************************************************************************************************************************************//
	//***************************************************************************************************************************************************************************************************//		
	generate_parameter_value_test_comparison_data 						= true;
	generate_parameter_value_test_RSP_comparison_data 					= true;
	generate_parameter_value_test_RSP_error_comparison_data 			= true;
	generate_parameter_value_test_std_dev_comparison_data 				= true;
	generate_parameter_value_test_TV_comparison_data	 				= true;				
	//***************************************************************************************************************************************************************************************************//
	//************* Parameter value test Booleans ************************************************************************************************************************************//
	//***************************************************************************************************************************************************************************************************//	
	perform_test_comparisons											= true;	
		perform_missing_analyses										= true;
		perform_TV_comparisons											= true;		
			write_TV_comparison_data 									= true;	
				overwrite_TV_comparison_data 							= true;	
			perform_TV_vs_iteration_analysis							= true;					
				create_TV_vs_iteration_plots							= true;		
					write_TV_vs_iteration_plots							= true;
						overwrite_TV_vs_iteration_plots					= true;
			perform_TV_vs_target_parameter_analysis						= true;	
				create_TV_vs_target_parameter_plots						= true;				
					write_TV_vs_target_parameter_plots					= true;
						overwrite_TV_vs_target_parameter_plots			= true;						
		perform_RSP_comparisons											= true;
			write_RSP_comparison_data									= true;
				overwrite_RSP_comparison_data							= true;
			create_RSP_comparison_plots									= true;
				write_RSP_comparison_plots								= true;
					overwrite_RSP_comparison_plots						= true;
		perform_RSP_error_comparisons 									= true;
			write_RSP_error_comparison_data								= true;
				overwrite_RSP_error_comparison_data						= true;
			create_RSP_error_comparison_plots							= true;
				write_RSP_error_comparison_plots						= true;
					overwrite_RSP_error_comparison_plots				= true;
		perform_std_dev_comparisons										= true;
			write_std_dev_comparison_data								= true;
				overwrite_std_dev_comparison_data						= true;
			create_std_dev_comparison_plots								= true;
				write_std_dev_comparison_plots							= true;
					overwrite_std_dev_comparison_plots					= true;	
	//***************************************************************************************************************************************************************************************************//
	//************* Multiplotting Booleans ************************************************************************************************************************************//
	//***************************************************************************************************************************************************************************************************//
	perform_test_comparisons											= true;	
	perform_test_comparisons_by_multiplot_parameter_combination			= true;	
	perform_test_comparisons_by_quality_measure							= false;	
		analyze_all_parameter_test_images								= false;
		perform_missing_analyses										= false;
		generate_RSP_multiplots 										= true;			
			write_RSP_multiplots										= true;
				overwrite_RSP_multiplots								= true;
			generate_RSP_multiplot_animations							= true;
				write_RSP_multiplot_animated_GIFs						= true;
					overwrite_RSP_multiplot_animated_GIFs				= true;
				write_RSP_multiplot_AVIs								= true;
					overwrite_RSP_multiplot_AVIs						= true;
		generate_RSP_error_multiplots 									= true;		
			write_RSP_error_multiplots									= true;
				overwrite_RSP_error_multiplots							= true;	
			generate_RSP_error_multiplot_animations						= true;
				write_RSP_error_multiplot_animated_GIFs					= true;
					overwrite_RSP_error_multiplot_animated_GIFs			= true;
				write_RSP_error_multiplot_AVIs							= true;
					overwrite_RSP_error_multiplot_AVIs					= true;
		generate_std_dev_multiplots 									= true;			
			write_std_dev_multiplots									= true;
				overwrite_std_dev_multiplots							= true;
			generate_std_dev_multiplot_animations						= true;
				write_std_dev_multiplot_animated_GIFs					= true;
					overwrite_std_dev_multiplot_animated_GIFs			= true;
				write_std_dev_multiplot_AVIs							= true;
					overwrite_std_dev_multiplot_AVIs					= true;
		generate_TV_multiplots											= true;
			plot_TV_multiplots 											= true;
				write_TV_multiplots 									= true;
					overwrite_TV_multiplots								= true;				
			plot_TV_step_multiplots 									= true;
				write_TV_step_multiplots								= true;
					overwrite_TV_step_multiplots						= true;				
			plot_TV_difference_multiplots 								= false;
				write_TV_difference_multiplots							= false;
					overwrite_TV_difference_multiplots					= false;				
		generate_TV_animations											= true;
			generate_TV_multiplot_animations							= true;
				write_TV_multiplot_animated_GIFs						= true;
					overwrite_TV_multiplot_animated_GIFs				= true;
				write_TV_multiplot_AVIs									= true;
					overwrite_TV_multiplot_AVIs							= true;
			generate_TV_step_multiplot_animations						= true;
				write_TV_step_multiplot_animated_GIFs					= true;
					overwrite_TV_step_multiplot_animated_GIFs			= true;
				write_TV_step_multiplot_AVIs							= false;
					overwrite_TV_step_multiplot_AVIs					= false;
			generate_TV_difference_multiplot_animations					= false;
				write_TV_difference_multiplot_animated_GIFs				= false;
					overwrite_TV_difference_multiplot_animated_GIFs		= false;
				write_TV_difference_multiplot_AVIs						= false;
					overwrite_TV_difference_multiplot_AVIs				= false;
	//***********************************************************************************************************************************************************************************************//
	//***************************************************************************** Set reconstruction data directory/file info *********************************************************************//
	//***********************************************************************************************************************************************************************************************//
	RSP_Boolean_indices 										= 0;
	RSP_error_Boolean_indices 									= 1;
	std_dev_Boolean_indices 									= 2;
	TV_Boolean_indices 											= 3;
	TV_step_Boolean_indices 									= 4;
	generate_multiplots_Boolean_indices 						= 0;
	write_multiplots_Boolean_indices 							= 1;
	overwrite_multiplots_indices 								= 2;
	write_multiplot_animated_GIFs_Boolean_indices 				= 3;
	overwrite_multiplot_animated_GIFs_Boolean_indices 			= 4;
	write_multiplot_AVIs_Boolean_indices  						= 5;
	overwrite_multiplot_AVIs_Boolean_indices  					= 6;	
	multiplot_Booleans 											= newArray(generate_RSP_multiplots, generate_RSP_error_multiplots, generate_std_dev_multiplots, plot_TV_multiplots, plot_TV_step_multiplots);
	write_multiplot_Booleans 									= newArray(write_RSP_multiplots, write_RSP_error_multiplots, write_std_dev_multiplots, write_TV_multiplots, write_TV_step_multiplots);
	overwrite_multiplot_Booleans 								= newArray(overwrite_RSP_multiplots, overwrite_RSP_error_multiplots, overwrite_std_dev_multiplots, overwrite_TV_multiplots, overwrite_TV_step_multiplots);
	write_multiplot_animated_GIF_Booleans 						= newArray(write_RSP_multiplot_animated_GIFs, write_RSP_error_multiplot_animated_GIFs, write_std_dev_multiplot_animated_GIFs, write_TV_multiplot_animated_GIFs, write_TV_step_multiplot_animated_GIFs);
	overwrite_multiplot_animated_GIF_Booleans					= newArray(overwrite_RSP_multiplot_animated_GIFs, overwrite_RSP_error_multiplot_animated_GIFs, overwrite_std_dev_multiplot_animated_GIFs, overwrite_TV_multiplot_animated_GIFs, overwrite_TV_step_multiplot_animated_GIFs);
	write_multiplot_AVI_Booleans 								= newArray(write_RSP_multiplot_AVIs, write_RSP_error_multiplot_AVIs, write_std_dev_multiplot_AVIs, write_TV_multiplot_AVIs, write_TV_step_multiplot_AVIs);
	overwrite_multiplot_AVI_Booleans 							= newArray(overwrite_RSP_multiplot_AVIs, overwrite_RSP_error_multiplot_AVIs, overwrite_std_dev_multiplot_AVIs, overwrite_TV_multiplot_AVIs, overwrite_TV_step_multiplot_AVIs);
	PNG_image_Booleans											= newArray(print_input_PNG_paths, close_input_PNG_images);
	animation_Booleans 											= newArray(close_multiplot_stack_images, print_multiplot_GIF_paths, print_multiplot_AVI_paths); 
	RSP_Booleans												= newArray(generate_RSP_multiplots, write_RSP_multiplots, overwrite_RSP_multiplots, write_RSP_multiplot_animated_GIFs, overwrite_RSP_multiplot_animated_GIFs, write_RSP_multiplot_AVIs, overwrite_RSP_multiplot_AVIs);
	RSP_error_Booleans 											= newArray(generate_RSP_error_multiplots, write_RSP_error_multiplots, overwrite_RSP_error_multiplots, write_RSP_error_multiplot_animated_GIFs, overwrite_RSP_error_multiplot_animated_GIFs, write_RSP_error_multiplot_AVIs, overwrite_RSP_multiplot_AVIs);
	std_dev_Booleans 											= newArray(generate_std_dev_multiplots, write_std_dev_multiplots, overwrite_std_dev_multiplots, write_std_dev_multiplot_animated_GIFs, overwrite_std_dev_multiplot_animated_GIFs, write_std_dev_multiplot_AVIs, overwrite_std_dev_multiplot_AVIs);
	TV_Booleans 												= newArray(plot_TV_multiplots, write_TV_multiplots, overwrite_TV_multiplots, write_TV_multiplot_animated_GIFs, overwrite_TV_multiplot_animated_GIFs, write_TV_multiplot_AVIs, overwrite_TV_multiplot_AVIs);
	TV_step_Booleans											= newArray(plot_TV_step_multiplots, write_TV_step_multiplots, overwrite_TV_step_multiplots, write_TV_step_multiplot_animated_GIFs, overwrite_TV_step_multiplot_animated_GIFs, write_TV_step_multiplot_AVIs, overwrite_TV_step_multiplot_AVIs);
	//***********************************************************************************************************************************************************************************************//
	//***************************************************************************** Set reconstruction data directory/file info *********************************************************************//
	//***********************************************************************************************************************************************************************************************//
	folder_separator								= File.separator;
	ImageJ_plugins_directory						= getDirectory("plugins");
	ImageJ_macro_directory							= getDirectory("macros");
	ImageJ_program_directory						= getDirectory("imagej") ;
	ImageJ_launch_directory							= getDirectory("startup");
	ImageJ_previous_macro							= getInfo("macro.filepath");
	github_macro_directory							= "C:\\Users\\Blake\\Documents\\GitHub\\Baylor_ICTHUS\\pCT_Reconstruction\\Tools\\ImageJ\\";
 	reconstruction_data_directory_C					= "C:\\Users\\Blake\\Documents\\Education\\Research\\pCT\\pCT_data\\reconstruction_data";
	reconstruction_data_directory_D					= "D:\\pCT\\pCT_data\\reconstruction_data";
	ROI_analysis_macro_filename 					= "ROI_Analysis.ijm";
	ROI_analysis_macro_path 						= github_macro_directory + ROI_analysis_macro_filename;
	multiplotting_macro_filename 					= "Multiplotting.ijm";
	multiplotting_macro_path 						= github_macro_directory + multiplotting_macro_filename;
	test_result_comparison_macro_filename 			= "Test_Result_Comparison.ijm";
	test_result_comparison_macro_path 				= github_macro_directory + test_result_comparison_macro_filename;
	reconstruction_data_folder						= folder_separator + "reconstruction_data";
	simulated_data_folder							= folder_separator + "Simulated";
	experimental_data_folder						= folder_separator + "Experimental";
	output_folder									= folder_separator + "Output";
	geant4_data_folder_basename						= folder_separator + "G_";
	phantom_basename								= "CTP404_Sensitom";
	//phantom_basename								= "CTP404_Sensitom_4M";
	phantom_name_folder								= folder_separator + phantom_basename;
	run_date										= "15-05-24";
	//run_date 										= "14-12-11";
	run_date_folder									= geant4_data_folder_basename + run_date;
	run_number_folder								= folder_separator + "0001";
	preprocess_date_folder							= folder_separator + "15-05-24";
	//preprocess_date								= folder_separator + "14-12-11";
	
	// Options controlling construction of reconstruction_data_directory/test_batch_directory
	simulated_data									= "Simulated";
	experimental_data								= "Experimental";	
	drive_C											= "C";
	drive_D											= "D";
	current_reconstruction_data_drive				= "D";
	current_reconstruction_data_type				= simulated_data;
	
	if(current_reconstruction_data_drive == drive_C)				reconstruction_data_directory 	= reconstruction_data_directory_C;
	else if(current_reconstruction_data_drive == drive_D)			reconstruction_data_directory 	= reconstruction_data_directory_D;
	if(current_reconstruction_data_type == simulated_data) 			test_batch_directory 			= reconstruction_data_directory + phantom_name_folder + simulated_data_folder + run_date_folder + run_number_folder + output_folder + preprocess_date_folder + folder_separator;
	else if(current_reconstruction_data_type == experimental_data)	test_batch_directory 			= reconstruction_data_directory + phantom_name_folder + experimental_data_folder + run_date_folder + run_number_folder + output_folder + preprocess_date_folder + folder_separator;		

	// Image/data filename and directory prefixes, file extensions, and results table column headings
	TXT 											= ".txt";
	CSV 											= ".csv";
	PNG 											= ".png";
	GIF 											= ".gif";
	AVI 											= ".avi";
	IJM 											= ".ijm";
	TRUE											= "true";
	FALSE											= "false";
	CLOSE_PNG										= true;
	DONT_CLOSE_PNG									= false;
	SHORTEN_STRINGS									= true;
	DONT_SHORTEN_STRINGS							= false;	
	PRINT_PATH										= true;
	DONT_PRINT_PATH									= false;	
	PRINTING_ON										= true;
	PRINTING_OFF									= false;	
	PRINT_MAJOR_SECTION								= true;
	PRINT_MINOR_SECTION								= false;	
	CLOSE_WINDOW									= true;
	DONT_CLOSE_WINDOW								= false;	
	add_operation									= "add";
	multiply_operation								= "multiply";
	x_direction										= "x";
	y_direction										= "y";	 					
	circular_ROI									= "circle";
	square_ROI										= "square";		
	no_compression									= "None";
	JPEG_compression								= "JPEG";
	PNG_compression									= "PNG";
	ROI_definition_numbers_decoding_op				= "parseFloat";	
	ROI_definition_strings_decoding_op				= "none";	
	ROI_material_RSP_column_label 					= "Predicted RSP";
	RSP_error_column_label 							= "% Error";
	mean_column_label 								= "Mean";	
		
	// Input/Output directory/subdirectory names/prefixes 
	reconstructed_image_folder_prefix 				= "x";
	iteration_2_analyze_folder_prefix 				= "Iteration_";
	ROI_analysis_slice_2_analyze_folder_prefix 		= "Slice_";
	slice_2_analyze_folder_prefix 					= "slice_";
	slice_2_analyze_folder_short_prefix 			= "s";
	ROI_selection_diameter_folder_prefix 			= "d";
	compared_data_folder_suffix						= "_compared";
	averaged_data_folder_suffix						= "_avg";

	// Input/output info and data basenames/filenames
	ROI_definitions_filename_suffix					= "_ROIs" + TXT;	
	ROI_definitions_filename						= phantom_basename + ROI_definitions_filename_suffix;
	ROI_definitions_file_path						= github_macro_directory + folder_separator + ROI_definitions_filename;
	reconstructed_image_file_basenames 				= "x_";
	reconstructed_image_file_short_basenames 		= "x";
	initial_iterate_filename 						= reconstructed_image_file_basenames + "0" + TXT;	
	auto_break_filename 							= "autobreak.txt";
	auto_break_path 								= ImageJ_macro_directory + auto_break_filename;
	parameter_test_info_basename					= "Test_Parameters_";
	specific_data_folders_filename 					= "ROI_analysis_folders.txt";
	reconstruction_folders_filename 				= "reconstruction_folders.txt";
	averaging_folders_filename 						= "averaging_folders.txt";
	multiplot_folders_filename 						= "multiplot_folders.txt";
	compared_folders_filename 						= "compared_folders.txt";	
	initial_iterate_filename 						= reconstructed_image_file_basenames + "0" + TXT;	
	profile_CSV_basename 							= "Line_Profiles";
	profile_data_file_basenames 					= "Line_Profile";
	regions_data_file_basenames						= "Region_Statistics";
	gradient_CSV_basename 							= "Gradients";
	gradient_data_file_basenames 					= "Gradient";	
	RSP_data_file_basenames							= "RSP";
	RSP_error_data_file_basenames					= "RSP_error";
	std_dev_data_file_basenames						= "Std_Dev";
	TV_data_file_basenames							= "TV";
	TV_measurements_basename						= "TV_measurements";
	TV_step_data_file_basenames						= "TV_step";
	TV_difference_data_file_basenames				= "TV_difference";	
	
	RSP_data_file_base_nicknames					= "RSP";
	RSP_error_data_file_base_nicknames				= "RSPe";
	std_dev_data_file_base_nicknames				= "SD";
	TV_data_file_base_nicknames						= "TV";
	TV_measurements_base_nickname					= "TV";
	TV_step_data_file_base_nicknames				= "TVs";
	TV_difference_data_file_base_nicknames			= "TVd";	
	
	RSP_output_basenames							= "RSP";
	RSP_error_output_basenames						= "RSP_error";
	std_dev_output_basenames						= "Std_Dev";
	TV_output_basenames								= "TV";
	TV_step_output_basenames						= "TV_step";
	TV_difference_output_basenames					= "TV_difference";	
	
	RSP_output_short_basenames						= "RSP";
	RSP_error_output_short_basenames				= "RSPe";
	std_dev_output_short_basenames					= "SD";
	TV_output_short_basenames						= "TV";
	TV_step_output_short_basenames					= "TVs";
	TV_difference_output_short_basenames			= "TVd";	
	
	ROI_analysis_RSP_output_basename 				= "RSP";
	ROI_analysis_RSP_error_output_basename 			= "RSP_error";
	ROI_analysis_std_dev_output_basename 			= "std_dev";
	ROI_analysis_TV_output_basename 				= "TV_measurements";
	parameter_value_test_RSP_output_basename 		= "RSP";
	parameter_value_test_RSP_error_output_basename 	= "RSP_error";
	parameter_value_test_std_dev_output_basename 	= "std_dev";
	parameter_value_test_TV_output_basename 		= "TV_measurements";
	multiplot_RSP_output_basename					= "RSP";
	multiplot_RSP_error_output_basename				= "RSP_error";
	multiplot_std_dev_output_basename				= "std_dv";
	multiplot_TV_output_basename					= "RSP";
	//***********************************************************************************************************************************************************************************************//
	//************************************* Import ROI definitions for current phantom and set corresponding internal variables/arrays **************************************************************//
	//***********************************************************************************************************************************************************************************************//
	ROI_parameter_decodings						= newArray(ROI_definition_strings_decoding_op, ROI_definition_strings_decoding_op, ROI_definition_strings_decoding_op, ROI_definition_strings_decoding_op, ROI_definition_numbers_decoding_op, ROI_definition_numbers_decoding_op, ROI_definition_numbers_decoding_op, ROI_definition_strings_decoding_op);		
	ROI_definitions_parameter_list				= newArray("ROI_material_names", "ROI_labels", "ROI_label_nicknames", "ROI_shapes", "ROI_diameters", "ROI_selection_radii", "ROI_profile_radius_beyond_ROI", "bulk_material" );
	ROI_parameter_strings						= file_2_key_value_pairs(github_macro_directory, ROI_definitions_filename, phantom_basename, 	ROI_definitions_parameter_list, ROI_parameter_decodings, ROI_definition_numbers_decoding_op, print_ROI_definitions_path);		
	ROI_material_names 							= ROI_parameter_string_2_values(ROI_definitions_parameter_list[0],  ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);	
	ROI_labels 									= ROI_parameter_string_2_values(ROI_definitions_parameter_list[1], 	ROI_definitions_parameter_list, ROI_parameter_strings,  	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);
	ROI_label_nicknames 						= ROI_parameter_string_2_values(ROI_definitions_parameter_list[2], 	ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);					//bulk_material = bulk_material[0];
	ROI_shapes 									= ROI_parameter_string_2_values(ROI_definitions_parameter_list[3], 	ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);	
	ROI_diameters 								= ROI_parameter_string_2_values(ROI_definitions_parameter_list[4],	ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);
	ROI_selection_radii 						= ROI_parameter_string_2_values(ROI_definitions_parameter_list[5], 	ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);
	ROI_profile_radius_beyond_ROI	 			= ROI_parameter_string_2_values(ROI_definitions_parameter_list[6],	ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);	//ROI_profile_radius_beyond_ROI = ROI_profile_radius_beyond_ROI[0];
	bulk_material 								= ROI_parameter_string_2_values(ROI_definitions_parameter_list[7], 	ROI_definitions_parameter_list, ROI_parameter_strings, 	ROI_parameter_decodings, ROI_definition_numbers_decoding_op);					//bulk_material = bulk_material[0];
	bulk_material_RSP 							= material_name_2_RSP(bulk_material, simulated_scan);
	ROI_material_RSPs 							= ROI_material_names_2_RSPs(ROI_material_names, simulated_scan);
	num_ROIs_2_analyze 							= ROI_material_names.length; 										// # of material ROI ROIs in phantom
	//***********************************************************************************************************************************************************************************************//
	//************************************************************************************** Parameter value arrays *********************************************************************************//
	//***********************************************************************************************************************************************************************************************//
	empty_array 										= newArray();								// Passed to set_plot_extrema when no additional lines are plotted
	sequential_values									= Array.getSequence(100);					// Long sequential values array which other sequence arrays can be sliced from	
	num_specs 											= 4;										// # of test parameter properties specified in Specs file
	voxel_width 										= 1;										// 
	voxel_height										= 1;
	voxel_thickness										= 2.5;	
	voxel_dimensions									= newArray(voxel_width, voxel_height, voxel_thickness);
	voxels_per_mm										= 1;			
	flip_horizontally									= false;																// Specify if correct orientation of reconstructed images requires flipping horizontally
	flip_vertically										= true;																	// Specify if correct orientation of reconstructed images requires flipping vertically
	num_recon_iterations 								= 12;										// # of iterations of feasibility seeking performed in reconstruction
	images_per_reconstruction 							= num_recon_iterations + 1;					// 	
	first_iteration_2_analyze							= 0;
	last_iteration_2_analyze							= 12;
	iterations_2_analyze								= Array.slice(sequential_values, first_iteration_2_analyze, last_iteration_2_analyze + 1);
	num_iterations_2_analyze							= iterations_2_analyze.length;				// # of iterations of feasibility seeking image results to analyze
	iterations_2_analyze_string_precision 				= 0;										// # of digits after decimal point to use in conversion of iteration # to string 	
	//first_slice_2_analyze 							= x_slices/2;												// first slice analyzed by pCT_Analysis macro
	//last_slice_2_analyze 								= x_slices/2;												// last slice analyzed by pCT_Analysis macro		
	first_slice_2_analyze								= 10;
	last_slice_2_analyze								= 10;
	slices_2_analyze									= Array.slice(sequential_values, first_slice_2_analyze, last_slice_2_analyze + 1);
	num_slices_2_analyze								= slices_2_analyze.length;					// 
	slices_2_analyze_string_precision 					= 0;										// # of digits after decimal point to use in conversion of slice # to string 	
	ROI_radii 											= newArray(ROI_diameters.length);						// radii of circular selections used to analyze phantom ROIs
	ROI_profile_radii 									= newArray(ROI_diameters.length);							// Set distance to extend profile line left/right from material insert ROI centers
	ROI_selection_radii 								= newArray(3.5, 4, 6);						// radii of circular selections used to analyze phantom ROIs
	ROI_selection_diameters 							= newArray(ROI_selection_radii.length);		// diameters of circular selections used to analyze phantom ROIs	
	ROI_selection_diameter_string_precision				= 0;										// # of digits after decimal point to use in conversion of an ROI selection diameter value to string	
	num_ROI_selection_diameters							= ROI_selection_diameters.length;			// diameters of circular selections used to analyze phantom ROIs 
	//***********************************************************************************************************************************************************************************************//
	//************************* Construct commonly used strings for parameter values and files/folders **********************************************************************************************//
	//***********************************************************************************************************************************************************************************************//	
	reconstructed_image_strings 						= newArray(images_per_reconstruction);		
	reconstructed_image_short_strings 					= newArray(images_per_reconstruction);		
	reconstructed_image_folders 						= newArray(images_per_reconstruction);		
	reconstructed_image_short_folders 					= newArray(images_per_reconstruction);		
	reconstructed_image_filenames 						= newArray(images_per_reconstruction);		
	iterations_2_analyze_strings 						= newArray(images_per_reconstruction);		
	iterations_2_analyze_folders 						= newArray(images_per_reconstruction);		
	slices_2_analyze_strings 							= newArray(num_slices_2_analyze);						
	slices_2_analyze_folders 							= newArray(num_slices_2_analyze);						
	ROI_analysis_slices_2_analyze_folders 				= newArray(num_slices_2_analyze);						
	slices_2_analyze_short_strings 						= newArray(num_slices_2_analyze);						
	slices_2_analyze_short_folders 						= newArray(num_slices_2_analyze);						
	ROI_selection_diameter_strings 						= newArray(num_ROI_selection_diameters);				
	ROI_selection_diameter_folders 						= newArray(num_ROI_selection_diameters);
	for(i = 0; i <= num_recon_iterations; i++)	
	{							
		reconstructed_image_strings[i] 					= reconstructed_image_file_basenames + d2s(i, iterations_2_analyze_string_precision);		
		reconstructed_image_folders[i] 					= folder_separator +  reconstructed_image_file_basenames + d2s(i, iterations_2_analyze_string_precision);	
		reconstructed_image_short_strings[i] 			= reconstructed_image_file_short_basenames + d2s(i, iterations_2_analyze_string_precision);		
		reconstructed_image_short_folders[i] 			= folder_separator +  reconstructed_image_file_short_basenames + d2s(i, iterations_2_analyze_string_precision);	
		reconstructed_image_filenames[i] 				= reconstructed_image_file_basenames + d2s(i, iterations_2_analyze_string_precision) + TXT;		
		iterations_2_analyze_strings[i] 				= d2s(i, iterations_2_analyze_string_precision);		
		iterations_2_analyze_folders[i] 				= folder_separator + iteration_2_analyze_folder_prefix + d2s(i, iterations_2_analyze_string_precision);	
	}	
	for(i = 0; i < num_slices_2_analyze; i++)	
	{							
		slices_2_analyze_strings[i] 					= d2s(slices_2_analyze[i], slices_2_analyze_string_precision);		
		slices_2_analyze_folders[i] 					= folder_separator + slice_2_analyze_folder_prefix + d2s(slices_2_analyze[i], slices_2_analyze_string_precision);	
		slices_2_analyze_short_strings[i] 				= d2s(slices_2_analyze[i], slices_2_analyze_string_precision);		
		slices_2_analyze_short_folders[i] 				= folder_separator + slice_2_analyze_folder_short_prefix + d2s(slices_2_analyze[i], slices_2_analyze_string_precision);	
		ROI_analysis_slices_2_analyze_folders[i]		= folder_separator + ROI_analysis_slice_2_analyze_folder_prefix + d2s(slices_2_analyze[i], slices_2_analyze_string_precision);	
	}
	for(i = 0; i < num_ROI_selection_diameters; i++)								
	{																					
		ROI_selection_diameters[i] 						= 2 * ROI_selection_radii[i];									
		ROI_selection_diameter_strings[i] 				= d2s(ROI_selection_diameters[i], ROI_selection_diameter_string_precision);	
		ROI_selection_diameter_folders[i] 				= folder_separator + ROI_selection_diameter_folder_prefix + d2s(ROI_selection_diameters[i], ROI_selection_diameter_string_precision);		
	}																								
	for(i = 0; i < ROI_diameters.length; i++)
	{
		ROI_radii[i] 									= ROI_diameters[i] / 2;		
		ROI_profile_radii[i]							= ROI_radii[i] + ROI_profile_radius_beyond_ROI;	
	}			
	//reconstructed_image_range_string					= "[" + reconstructed_image_strings[0] + "-" + reconstructed_image_strings[last_iteration_2_analyze] + "]";
	reconstructed_image_range_string					= reconstructed_image_file_short_basenames + "[" + iterations_2_analyze_strings[1] + "-" + iterations_2_analyze_strings[last_iteration_2_analyze] + "]";
	iterations_2_analyze_range_string					= "[" + iterations_2_analyze_strings[first_iteration_2_analyze] + "-" + iterations_2_analyze_strings[last_iteration_2_analyze] + "]";
	//***********************************************************************************************************************************************************************************************//
	//************************* Construct commonly used strings for parameter values and files/folders **********************************************************************************************//
	//***********************************************************************************************************************************************************************************************//			
	num_specs 									= 4;											// # of test parameter properties specified in Specs file
	tolerance									= 0.0000001;								// Tolerance separating min/max values of array used in findMinima/findMaxima
	filterwidth 								= 3;											// Width of filter used in median filtering FBP image
	sobel_x 									= "[-1 0 1\n-2 0 2\n-1 0 1\n]";					// Matrix convolved with image to generate approximate image gradient in x-direction
	sobel_y 									= "[-1 -2 -1\n0 0 0\n1 2 1\n]";					// Matrix convolved with image to generate approximate image gradient in y-direction
	scharr_x 									= "[-3 0 3\n-10 0 10\n-3 0 3\n]";				// Matrix convolved with image to generate approximate image gradient in x-direction
	scharr_y 									= "[-3 -10 -3\n0 0 0\n3 10 3\n]";				// Matrix convolved with image to generate approximate image gradient in y-direction
	gradient_x_kernel							= scharr_x;										// Specify which matrix to convolve with image to generate approximate x-direction derivative
	gradient_y_kernel							= scharr_y;										// Specify which matrix to convolve with image to generate approximate x-direction derivative
	x_magnification 							= 5;											// Factor by which extracted slices are magnified before saving as PNG
	lower_limits_scale 							= 0.99;										// Scale factor to apply to minimum value of curve to define lower axis limit
	upper_limits_scale 							= 1.01;										// Scale factor to apply to maximum value of curve to define upper axis limit
	difference_scale							= 0.1;										// Scale factor to apply to minimum/maximum value of y-axis to improve appearance
	alignment_column 							= 25;
	screen_width								= screenWidth;
	screen_height								= screenHeight;
	x_frame_size								= 480;										// Size of plots (in pixels) in x-direction
	y_frame_size								= 380;										// Size of plots (in pixels) in y-direction
	//x_frame_size								= 1200;										// Size of plots (in pixels) in x-direction
	//y_frame_size								= 600;										// Size of plots (in pixels) in y-direction
	//text_xpos_ratio								= 0.5;										//
	//text_ypos_ratio								= 0.07;										//
	text_xpos_ratio								= 0.45;										//
	text_ypos_ratio								= 0.01;										//
	//text_size									= 40;
	//legend_text_size							= 22;
	//axis_label_text_size						= 34;
	text_size									= 30;
	legend_text_size							= 22;
	axis_label_text_size						= 28;
	data_line_color								= "blue";										// Specify color of primary plot curves
	add_line_color								= "green";										// Specify color of secondary plot curves
	color_rotation								= newArray("green", "blue",  "cyan",  "magenta", "orange", "red", "pink", "yellow", "darkGray", "gray", "lightGray", "black", "white"); 
	add_text_justification						= "center";									//
	linewidth									= 3.5;										//
	plot_parameters 							= newArray(voxel_width, tolerance, lower_limits_scale, upper_limits_scale, difference_scale, x_frame_size, y_frame_size, text_xpos_ratio, text_ypos_ratio, add_line_color, linewidth);
	grayscale_range_min							= 0.0;											// Specify min value of grayscale range, all values at or below are shown as black 
	grayscale_range_max							= 2.0;											// Specify max value of grayscale range, all values at or above are shown as white 
	GIF_frame_rate								= 4;											// Specify animated GIF's frame rate in frames per second (fps)
	animated_GIF_info							= newArray(print_multiplot_GIF_paths, close_stack_images, close_multiplot_stack_images, GIF_frame_rate );
	AVI_compression_format						= PNG_compression;								// Specify image compression format used in constructing AVI  video from image stack
	AVI_frame_rate								= 5;											// Specify AVI video frame rate in frames per second (fps)	
	AVI_info									= newArray(print_multiplot_AVI_paths, close_stack_images, close_multiplot_stack_images, AVI_frame_rate, AVI_compression_format );	
	make_top_directory							= true;
	dont_make_top_directory						= false;
	INPUT_FILE_LIST 							= newArray();   		
	OUTPUT_FILE_LIST							= newArray();
	COPIED_FILE_LIST							= newArray();
	COPIED_FILE_FROM_LIST						= newArray();
	COPIED_FILE_TO_LIST							= newArray();
	DIRECTORIES_CREATED							= newArray();
	debug_print_counter							= 1;
	//***********************************************************************************************************************************************************************************************************//
	//***************** Specifications used to identify file containing specifications of parameters of parameter value test ************************************************************************************//
	//***********************************************************************************************************************************************************************************************************//
	parameter_test_number						= 5;
	parameter_test_info_filename				= parameter_test_info_basename + d2s(parameter_test_number, 0)  + TXT;
	parameter_test_info 						= file_2_array(test_batch_directory, parameter_test_info_filename, print_input_data_path);
	//Ap(parameter_test_info);
	num_parameters 								= parameter_test_info.length;
	parameter_values 							= newArray();
	num_parameter_values 						= newArray(num_parameters);
	parameter_string_prefixes 					= newArray(num_parameters);
	parameter_string_short_prefixes 			= newArray(num_parameters);
	parameter_string_precisions 				= newArray(num_parameters);				
	for(i = 0; i < parameter_test_info.length; i++)
	{
		parameter_info 							= split(parameter_test_info[i]);
		parameter_string_prefixes[i] 			= parameter_info[0];
		parameter_string_short_prefixes[i] 		= parameter_info[1];
		parameter_string_precisions[i] 			= parseInt(parameter_info[2]);
		num_parameter_values[i] 				= parameter_info.length - 3;
		parameter_elements 						= Array.slice(parameter_info, 3, parameter_info.length);
		parameter_values 						= Array.concat(parameter_values, parameter_elements);		
	}
	//for(i = 0; i < parameter_values.length; i++)
	//	parameter_values[i] 					= parseFloat(parameter_values[i]);	
	parameter_value_offsets 					= generate_parameter_value_offsets(num_parameter_values);
	parameter_value_strings 					= generate_all_parameter_value_strings(parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
	modulo_values 								= generate_modulo_values(num_parameter_values);
	total_combinations 							= series_product(num_parameter_values);
	//***********************************************************************************************************************************************************************************************//
	//************************************************************************************** Parameter value test  *********************************************************************************//
	//***********************************************************************************************************************************************************************************************//	
	target_test_parameter						= num_parameters;
	target_test_parameter_index					= num_parameters - 1;
	target_test_parameter_prefix				= parameter_string_prefixes[target_test_parameter_index];
	target_test_parameter_values 				= extract_parameter_values(target_test_parameter_index, parameter_values, num_parameter_values, parameter_value_offsets);		
	target_test_parameter_value_strings 		= generate_parameter_value_strings(target_test_parameter_values, parameter_string_precisions[target_test_parameter_index], false);
	target_test_parameter_value_short_strings 	= generate_parameter_value_strings(target_test_parameter_values, parameter_string_precisions[target_test_parameter_index], true);
	target_test_parameter_equals_strings		= parameter_values_2_parameter_equals_strings(target_test_parameter_values, target_test_parameter_prefix, parameter_string_precisions[target_test_parameter_index]);
	target_test_parameter_equals_short_strings	= parameter_values_2_parameter_equals_short_strings(target_test_parameter_values, target_test_parameter_prefix, parameter_string_precisions[target_test_parameter_index]);
	target_test_parameter_min_value				= target_test_parameter_values[0];
	target_test_parameter_max_value				= target_test_parameter_values[target_test_parameter_values.length - 1];
	target_test_parameter_min_value_string		= target_test_parameter_value_short_strings[0];
	target_test_parameter_max_value_string		= target_test_parameter_value_short_strings[target_test_parameter_values.length - 1];
	target_test_parameter_range_suffix			=  "["	+ target_test_parameter_min_value_string + "-" + target_test_parameter_max_value_string + "]";			// e.g. [1-12]
	target_test_parameter_range_filenaming 		= parameter_string_prefixes[target_test_parameter_index] 	+ 			target_test_parameter_range_suffix;	// e.g. L0_0_Nk[1-12]	
	target_test_parameter_range_plots 			= parameter_string_prefixes[target_test_parameter_index] 	+ " = " + 	target_test_parameter_range_suffix;		// e.g. L0_0_Nk = [1-12]		
	num_target_test_parameter_values 			= num_parameter_values[target_test_parameter_index];
	num_target_test_parameter_tests 			= total_combinations / num_target_test_parameter_values;	
	//***********************************************************************************************************************************************************************************************//
	//************************************************************************************** Parameter value test  *********************************************************************************//
	//***********************************************************************************************************************************************************************************************//	
	multiplot_parameter_prefix					= "A";																							// Specifying multiplot target parameter using its parameter string prefix
	multiplot_parameter_index					= parameter_string_2_index(parameter_string_prefixes, multiplot_parameter_prefix);		// Multiplot target parameter index (from 0 - N-1)used to access elements of the parameter values/strings/precisions arrays
	multiplot_parameter							= multiplot_parameter_index + 1;															// Multiplot target parameter # from 1-N, not it's array index (i.e. from 0 - N-1)	
	multiplot_parameter_values					= extract_parameter_values(multiplot_parameter_index, parameter_values, num_parameter_values, parameter_value_offsets);
	multiplot_parameter_value_strings 			= generate_parameter_value_strings(multiplot_parameter_values,  parameter_string_precisions[multiplot_parameter_index], false);
	multiplot_parameter_value_short_strings 	= generate_parameter_value_strings(multiplot_parameter_values,  parameter_string_precisions[multiplot_parameter_index], true);
	multiplot_parameter_equals_strings			= parameter_values_2_parameter_equals_strings(multiplot_parameter_values, multiplot_parameter_prefix, parameter_string_precisions[multiplot_parameter_index]);
	multiplot_parameter_equals_short_strings	= parameter_values_2_parameter_equals_short_strings(multiplot_parameter_values, multiplot_parameter_prefix, parameter_string_precisions[multiplot_parameter_index]);
	multiplot_test_parameter_min_value			= multiplot_parameter_values[0];
	multiplot_test_parameter_max_value			= multiplot_parameter_values[multiplot_parameter_values.length - 1];
	multiplot_parameter_min_value_string		= multiplot_parameter_value_short_strings[0];
	multiplot_parameter_max_value_string		= multiplot_parameter_value_short_strings[multiplot_parameter_values.length - 1];
	multiplot_parameter_range_suffix			=  "["	+ multiplot_parameter_min_value_string + "-" + multiplot_parameter_max_value_string + "]";	// e.g.[0-1]
	multiplot_parameter_range_filenaming 		= parameter_string_prefixes[multiplot_parameter_index] + multiplot_parameter_range_suffix;		;	// e.g.TV[0-1]
	multiplot_parameter_range_plots	 			= parameter_string_prefixes[multiplot_parameter_index] + " = " + multiplot_parameter_range_suffix;	// e.g. TV = [0-1]	
	multiplot_animations_first_iteration		= 1;
	multiplot_animations_last_iteration			= 12;
	multiplot_animations_image_range_string		= "[" + reconstructed_image_strings[multiplot_animations_first_iteration] + "-" + reconstructed_image_strings[multiplot_animations_last_iteration] + "]";
	multiplot_animations_iteration_range_string	= "[" + iterations_2_analyze_strings[multiplot_animations_first_iteration] + "-" + iterations_2_analyze_strings[multiplot_animations_last_iteration] + "]";	
	num_multiplot_parameter_values				= num_parameter_values[multiplot_parameter_index];			
	reduced_target_test_parameter_index			= target_test_parameter_index - 1;
	reduced_modulo_values 						= generate_reduced_modulo_values(num_parameter_values, modulo_values, multiplot_parameter_index);
	reduced_parameter_value_offsets 			= generate_reduced_parameter_value_offsets(num_parameter_values, parameter_value_offsets, multiplot_parameter_index);	
	reduced_total_combinations 					= total_combinations / num_multiplot_parameter_values;
	//***********************************************************************************************************************************************************************************************//
	//** Indivudal parameter value analysis and comparison ******************************************************************************************************************************************//
	//***********************************************************************************************************************************************************************************************//
	//comparison_parameter_prefix					= "B";																							// Specifying multiplot target parameter using its parameter string prefix
	//comparison_vs_parameter_prefix				= "L";																							// Specifying multiplot target parameter using its parameter string prefix
	//comparison_parameter_index					= parameter_string_2_index(parameter_string_prefixes, comparison_parameter_prefix);																							// Specifying multiplot target parameter using its parameter string prefix
	//comparison_vs_parameter_index				= parameter_string_2_index(parameter_string_prefixes, comparison_vs_parameter_prefix);		// Multiplot target parameter index (from 0 - N-1)used to access elements of the parameter values/strings/precisions arrays
	//comparison_parameter						= comparison_parameter_index + 1;															// Multiplot target parameter # from 1-N, not it's array index (i.e. from 0 - N-1)	
	//comparison_vs_parameter						= comparison_vs_parameter_index + 1;															// Multiplot target parameter # from 1-N, not it's array index (i.e. from 0 - N-1)	
	//comparison_parameter_values					= extract_parameter_values(comparison_parameter_index, parameter_values, num_parameter_values, parameter_value_offsets);
	//comparison_parameter_value_strings 			= generate_parameter_value_strings(comparison_parameter_values,  parameter_string_precisions[comparison_parameter_index], false);
	//comparison_parameter_value_short_strings 	= generate_parameter_value_strings(comparison_parameter_values,  parameter_string_precisions[comparison_parameter_index], true);
	//comparison_parameter_equals_strings			= parameter_values_2_parameter_equals_strings(comparison_parameter_values, comparison_parameter_prefix, parameter_string_precisions[comparison_parameter_index]);
	//comparison_parameter_equals_short_strings	= parameter_values_2_parameter_equals_short_strings(comparison_parameter_values, comparison_parameter_prefix, parameter_string_precisions[comparison_parameter_index]);
	////Ap(comparison_parameter_values);
	////Ap(comparison_parameter_value_strings);
	////Ap(comparison_parameter_value_short_strings);
	////Ap(comparison_parameter_equals_strings);
	////Ap(comparison_parameter_equals_short_strings);
	//num_comparison_parameter_values				= num_parameter_values[comparison_parameter_index];			
	//num_comparison_vs_parameter_values			= num_parameter_values[comparison_vs_parameter_index];			
	//for(i=0; i < num_comparison_parameter_values; i++)
	//{
	//	for(j=0; j < num_comparison_vs_parameter_values; j++)
	//	{
	//		
	//	}	
	//}
	
	
	////comparison_parameter_modulo_values 			= generate_reduced_modulo_values(num_parameter_values, modulo_values, comparison_parameter_index);
	//Ap(comparison_parameter_modulo_values);
	//comparison_parameter_value_offsets 			= generate_reduced_parameter_value_offsets(num_parameter_values, parameter_value_offsets, comparison_parameter_index);	
	//Ap(comparison_parameter_value_offsets);
	//comparison_total_combinations 				= total_combinations / num_comparison_parameter_values;
	//comparison_vs_total_combinations 			= total_combinations / num_comparison_vs_parameter_values;
	//exit();
	//***********************************************************************************************************************************************************************************************//
	//** For each parameter test value combination, determine path/folder of reconstruction results value *******************************************************************************************//
	//***********************************************************************************************************************************************************************************************//
	parameter_value_test_identifier 				= generate_combined_parameter_value_range_string(parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_prefixes, parameter_string_precisions);
	//parameter_value_tests_parent_directory				= test_batch_directory;
	//parameter_value_test_input_directory				= test_batch_directory;
	//parameter_value_test_output_directory				= test_batch_directory;
	parameter_value_tests_input_directory_parent	= test_batch_directory;
	parameter_value_tests_output_directory_parent	= test_batch_directory;	
	//current_parameter_value_test_output_directory 	= construct_valid_directory_path(test_batch_directory, parameter_value_test_identifier);
	current_parameter_value_test_output_directory_parent 	= construct_valid_directory_path(test_batch_directory, parameter_value_test_identifier);
	directory_created_successfully					= make_directory_recorded(current_parameter_value_test_output_directory_parent, DIRECTORIES_CREATED, print_directories_created);
	//parameter_test_directory 						= construct_valid_directory_path(test_batch_directory, parameter_value_test_identifier);
	//make_directory_recorded						(parameter_test_directory, DIRECTORIES_CREATED, print_directories_created);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//***********************************************************************************************************************************************************************************************//
	//** For each parameter test value combination, determine path/folder of reconstruction results value *******************************************************************************************//
	//***********************************************************************************************************************************************************************************************//
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	parameter_combination 						= newArray		(num_parameters);
	parameter_combination_indices 				= newArray		(num_parameters);
	parameter_strings 							= newArray		(num_parameters);
	reduced_parameter_combination_indices		= newArray		(num_parameters - 1);
	reduced_parameter_combination				= newArray		(num_parameters - 1);
	reduced_parameter_values_lower 				= Array.slice	(parameter_values, 0, parameter_value_offsets[multiplot_parameter_index]					);
	reduced_parameter_values_upper 				= Array.slice	(parameter_values,  parameter_value_offsets[multiplot_parameter], parameter_values.length);
	reduced_parameter_values 					= Array.concat	(reduced_parameter_values_lower, reduced_parameter_values_upper									);	
	all_parameter_combination_indices 			= newArray		();
	all_parameter_combinations 					= newArray		();
	all_folder_strings 							= newArray		(total_combinations);
	all_path_strings 							= newArray		(total_combinations);	
	all_reduced_parameter_combination_indices 	= newArray		();	    
	all_reduced_parameter_combinations 			= newArray		();
	all_reduced_folder_strings 					= newArray		(total_combinations);
	all_reduced_path_strings 					= newArray		(total_combinations);
	//exit();	
	for(combination_index = 0; combination_index < total_combinations; combination_index++)
	{
		//*******************************************************************************************************************************************************************************************//
		//***************************** Generate standard parameter value combinations and folders/paths ordered by target test parameter ***********************************************************//
		//*******************************************************************************************************************************************************************************************//
		parameter_combination_indices		 		= generate_parameter_test_combination_indices(target_test_parameter_index, combination_index, modulo_values, parameter_value_offsets );
		parameter_combination 						= generate_parameter_test_combination(parameter_combination_indices, parameter_values);
		all_parameter_combination_indices 			= Array.concat(all_parameter_combination_indices, parameter_combination_indices); 	
		all_parameter_combinations 					= Array.concat(all_parameter_combinations, parameter_combination); 	
		folder_string								= generate_parameter_test_folder(parameter_string_prefixes, parameter_string_precisions, parameter_combination );		
		all_folder_strings[combination_index] 		= folder_string;		
		all_path_strings[combination_index] 		= test_batch_directory + all_folder_strings[combination_index];	
		//*******************************************************************************************************************************************************************************************//
		//********************** Generate multiplotting parameter value combinations and folders/paths ordered by multiplotting target parameter ****************************************************//
		//*******************************************************************************************************************************************************************************************//		
		if(combination_index < reduced_total_combinations)
		{
			reduced_parameter_combination_indices 			= generate_parameter_test_combination_indices(target_test_parameter_index - 1, combination_index, reduced_modulo_values, reduced_parameter_value_offsets );
			reduced_parameter_combination 					= generate_parameter_test_combination(reduced_parameter_combination_indices, reduced_parameter_values);			
			all_reduced_parameter_combination_indices		= Array.concat(all_reduced_parameter_combination_indices, reduced_parameter_combination_indices); 
			all_reduced_parameter_combinations 				= Array.concat(all_reduced_parameter_combinations, reduced_parameter_combination); 
			start_index 									= combination_index * num_multiplot_parameter_values;
			for(i = 0; i < num_multiplot_parameter_values; i++)
			{
				folder_string 								= generate_parameter_test_multiplot_folder(multiplot_parameter_index, reduced_parameter_combination, i, parameter_values, parameter_value_offsets, parameter_string_prefixes, parameter_string_precisions );							
				all_reduced_folder_strings[start_index + i] = folder_string;
				all_reduced_path_strings[start_index + i] = test_batch_directory + folder_string;	
			}		
		}				
	} // END for( combination_index = 0; combination_index < total_combinations; combination_index++)
	//exit();
	//*******************************************************************************************************************************************************************************************//
	//**** Parameter value test plot and multiplot data array sizing, indexing, and partitioning info used to identify, extract/collect, and plot subets of data ********************************//
	//*******************************************************************************************************************************************************************************************//	
	ROIs_per_reconstruction 						= num_ROIs_2_analyze * images_per_reconstruction;
	ROIs_per_test 									= ROIs_per_reconstruction * num_target_test_parameter_values;
	ROIs_per_multiplot_data 						= ROIs_per_test * num_multiplot_parameter_values;
	ROIs_per_test_iteration							= num_target_test_parameter_values * num_ROIs_2_analyze;									
	ROI_material_RSP_line_by_target_parameter		= newArray(num_target_test_parameter_values);
	before_TVS_index 								= 0;
	after_TVS_index 								= 1;
	TV_step_measurements_per_iteration				= 2;
	TV_step_measurements_per_reconstruction 		= num_recon_iterations * TV_step_measurements_per_iteration;
	TV_step_measurements_per_multiplot_curve 		= num_target_test_parameter_values * TV_step_measurements_per_iteration;
	TV_step_measurements_per_multiplot 				= TV_step_measurements_per_multiplot_curve * num_multiplot_parameter_values;
	TV_step_measurements_per_test 					= TV_step_measurements_per_reconstruction * num_target_test_parameter_values;					
	TV_measurements_per_reconstruction 				= TV_step_measurements_per_reconstruction 	/ TV_step_measurements_per_iteration;
	TV_measurements_per_multiplot_curve 			= TV_step_measurements_per_multiplot_curve 	/ TV_step_measurements_per_iteration;
	TV_measurements_per_multiplot 					= TV_step_measurements_per_multiplot	 	/ TV_step_measurements_per_iteration;
	TV_measurements_per_test 						= TV_step_measurements_per_test 			/ TV_step_measurements_per_iteration;						
	TV_measurement_plots_per_iteration 				= 1;
	TV_difference_measurements_per_reconstruction 	= TV_step_measurements_per_reconstruction 	/ TV_step_measurements_per_iteration;
	TV_difference_measurements_per_multiplot_curve 	= TV_step_measurements_per_multiplot_curve 	/ TV_step_measurements_per_iteration;
	TV_difference_measurements_per_multiplot 		= TV_step_measurements_per_multiplot	 	/ TV_step_measurements_per_iteration;
	TV_difference_measurements_per_test 			= TV_step_measurements_per_test 			/ TV_step_measurements_per_iteration;						
	num_input_directories							= lengthOf(all_reduced_path_strings);
	num_input_directories_per_multiplot				= num_multiplot_parameter_values * num_target_test_parameter_values;
	num_multiplot_analyses							= num_input_directories / num_input_directories_per_multiplot;
	//*******************************************************************************************************************************************************************************************//
	//**** Parameter value test plot and multiplot data array sizing, indexing, and partitioning info used to identify, extract/collect, and plot subets of data ********************************//
	//*******************************************************************************************************************************************************************************************//
	zero_line_by_target_parameter_values 			= newArray(num_target_test_parameter_values);					
	zero_line_by_iteration							= newArray(ROIs_per_reconstruction);
	zero_line_by_ROI								= newArray(num_ROIs_2_analyze);			
	Array.fill										(zero_line_by_target_parameter_values, 0);	
	Array.fill										(zero_line_by_iteration, 0);	
	Array.fill										(zero_line_by_ROI, 0);	
	ROI_material_RSP_line_by_target_parameter		= newArray(num_target_test_parameter_values);
	ROI_material_RSP_line_no_air					= Array.slice(ROI_material_RSPs, 2, num_ROIs_2_analyze);			
	ROI_material_RSPs_by_iteration					= newArray(ROIs_per_reconstruction);
	RSPs_by_ROI										= newArray(ROIs_per_reconstruction);
	RSPs_by_iteration								= newArray(ROIs_per_reconstruction);
	RSP_errors_by_ROI								= newArray(ROIs_per_reconstruction);
	RSP_errors_by_iteration							= newArray(ROIs_per_reconstruction);
	std_devs_by_ROI									= newArray(ROIs_per_reconstruction);
	std_devs_by_iteration							= newArray(ROIs_per_reconstruction);			
	TV_measurements_by_iteration					= newArray(ROIs_per_reconstruction);
	indices_4_ordering_data 						= generate_ordering_indices(num_recon_iterations, num_ROIs_2_analyze, images_per_reconstruction, num_target_test_parameter_values);
	//*******************************************************************************************************************************************************************************************//
	//**** Parameter value test plot and multiplot data array sizing, indexing, and partitioning info used to identify, extract/collect, and plot subets of data ********************************//
	//*******************************************************************************************************************************************************************************************//	
	ROI_analysis_TV_input_filename						= TV_measurements_basename 		+ TXT;	
	ROI_analysis_RSP_output_filename					= RSP_data_file_basenames 		+ CSV;
	ROI_analysis_RSP_error_output_filename				= RSP_error_data_file_basenames	+ CSV;
	ROI_analysis_std_dev_output_filename				= std_dev_data_file_basenames  	+ CSV;
	ROI_analysis_TV_output_filename						= TV_data_file_basenames		+ TXT;
	
	parameter_value_test_RSP_input_filename				= ROI_analysis_RSP_output_filename;
	parameter_value_test_RSP_error_input_filename		= ROI_analysis_RSP_error_output_filename;
	parameter_value_test_std_dev_input_filename			= ROI_analysis_std_dev_output_filename;
	parameter_value_test_TV_input_filename				= ROI_analysis_TV_output_filename;	
	parameter_value_test_RSP_output_filename			= RSP_data_file_basenames 			+ "_" + target_test_parameter_range_suffix + TXT;
	parameter_value_test_RSP_error_output_filename		= RSP_error_data_file_basenames 	+ "_" + target_test_parameter_range_suffix + TXT;
	parameter_value_test_std_dev_output_filename		= std_dev_data_file_basenames 		+ "_" + target_test_parameter_range_suffix + TXT;
	parameter_value_test_TV_output_filename				= TV_data_file_basenames 			+ "_" + target_test_parameter_range_suffix + TXT;
	parameter_value_test_RSP_output_filename2			= RSP_data_file_basenames 			+ "_" + target_test_parameter_range_filenaming + TXT;
	parameter_value_test_RSP_error_output_filename2		= RSP_error_data_file_basenames 	+ "_" + target_test_parameter_range_filenaming + TXT;
	parameter_value_test_std_dev_output_filename2		= std_dev_data_file_basenames 		+ "_" + target_test_parameter_range_filenaming + TXT;
	parameter_value_test_TV_output_filename2			= TV_data_file_basenames 			+ "_" + target_test_parameter_range_filenaming + TXT;
	
	multiplot_RSP_input_filename						= parameter_value_test_RSP_output_filename;
	multiplot_RSP_error_input_filename					= parameter_value_test_RSP_error_output_filename;
	multiplot_std_dev_input_filename					= parameter_value_test_std_dev_output_filename;
	multiplot_TV_input_filename							= parameter_value_test_TV_output_filename;	
	//multiplot_RSP_output_basename						= RSP_data_file_basenames			+ "_" + target_test_parameter_range_suffix + "_" + multiplot_parameter_range_filenaming;
	//multiplot_RSP_error_output_basename					= RSP_error_data_file_basenames		+ "_" + target_test_parameter_range_suffix + "_" + multiplot_parameter_range_filenaming;
	//multiplot_std_dev_output_basename					= std_dev_data_file_basenames		+ "_" + target_test_parameter_range_suffix + "_" + multiplot_parameter_range_filenaming;
	//multiplot_TV_output_basename						= TV_data_file_basenames			+ "_" + target_test_parameter_range_suffix + "_" + multiplot_parameter_range_filenaming;
	multiplot_RSP_output_basename						= RSP_data_file_basenames			+ "_" + target_test_parameter_range_filenaming;
	multiplot_RSP_error_output_basename					= RSP_error_data_file_basenames		+ "_" + target_test_parameter_range_filenaming;
	multiplot_std_dev_output_basename					= std_dev_data_file_basenames		+ "_" + target_test_parameter_range_filenaming;
	multiplot_TV_output_basename						= TV_data_file_basenames			+ "_" + target_test_parameter_range_filenaming;
	multiplot_RSP_output_basename2						= RSP_data_file_basenames			+ "_" + target_test_parameter_range_filenaming + "_";
	multiplot_RSP_error_output_basename2				= RSP_error_data_file_basenames		+ "_" + target_test_parameter_range_filenaming + "_";
	multiplot_std_dev_output_basename2					= std_dev_data_file_basenames		+ "_" + target_test_parameter_range_filenaming + "_";
	multiplot_TV_output_basename2						= TV_data_file_basenames			+ "_" + target_test_parameter_range_filenaming + "_";
	
	test_multiplot_RSP_output_basename						= RSP_data_file_base_nicknames			+ "_" + target_test_parameter_range_filenaming;
	test_multiplot_RSP_error_output_basename					= RSP_error_data_file_base_nicknames		+ "_" + target_test_parameter_range_filenaming;
	test_multiplot_std_dev_output_basename					= std_dev_data_file_base_nicknames		+ "_" + target_test_parameter_range_filenaming;
	test_multiplot_TV_output_basename						= TV_data_file_base_nicknames			+ "_" + target_test_parameter_range_filenaming;
	test_multiplot_TV_step_output_basename						= TV_step_data_file_base_nicknames			+ "_" + target_test_parameter_range_filenaming;
	test_multiplot_TV_difference_output_basename						= TV_difference_data_file_base_nicknames			+ "_" + target_test_parameter_range_filenaming;
	
	
	
	//*******************************************************************************************************************************************************************************************//
	//**** Parameter value test plot and multiplot data array sizing, indexing, and partitioning info used to identify, extract/collect, and plot subets of data ********************************//
	//*******************************************************************************************************************************************************************************************//	
	if(check_all_paths_existence)
		verify_paths_exist(all_path_strings, print_missing_paths);		
	if(log_printing)
	{				
		print_ImageJ_info();		
		if(printing_ROI_definitions)
			print_ROI_definitions						();
		if(printing_reconstructed_image_analysis_info)						
		{
			//print_reconstructed_image_analysis_info		();
			print_reconstructed_image_info				();
			print_ROI_analysis_info						();
		}
		if(printing_parameter_value_test_info)								
			print_target_test_parameter_info				();
		if(printing_multiplot_parameter_info)							
			print_multiplot_parameter_info				();
		if(printing_input_output_filenames)
			print_input_output_filenames				();
	}
	if(analyze_specific_data)
	{
		print_section_separator	("Performing ROI analysis of specific list of reconstruction data output folders", PRINT_MAJOR_SECTION);
		//print("Analyzing compared data from:n" + compared_folders[0]);
		//specific_data_folders_filename 				= "ROI_analysis_folders.txt";
		specific_data_folders = file_2_array(test_batch_directory, specific_data_folders_filename, DONT_PRINT_PATH);	
		//for(i = 0; i < 1; i++)
		always_perform_analyses = !only_perform_missing_analyses;
		for(i = 0; i < specific_data_folders.length; i++)
		{
			reconstructed_data_folder = specific_data_folders[i];
			print_section_separator	("Performing ROI analysis on: " + reconstructed_data_folder, PRINT_MINOR_SECTION);
			//current_parameter_value_test_folder	= reconstruction_data_folders[i];
			reconstruction_data_exists 			= verify_reconstruction_output(test_batch_directory, reconstructed_data_folder, reconstructed_image_filenames, ROI_analysis_TV_input_filename, DONT_PRINT_PATH);
			data_verified 						= verify_ROI_analysis_output_files(test_batch_directory, reconstructed_data_folder, ROI_analysis_RSP_output_filename, ROI_analysis_RSP_error_output_filename, ROI_analysis_std_dev_output_filename, ROI_analysis_TV_input_filename, ROI_analysis_slices_2_analyze_folders, ROI_selection_diameter_folders, DONT_PRINT_PATH);		
			if(reconstruction_data_exists && (data_verified || always_perform_analyses))
				runMacro		(ROI_analysis_macro_path, test_batch_directory + reconstructed_data_folder + folder_separator);
			//else
				
			autobreak();
		}
		//	runMacro		(ROI_analysis_macro_path, test_batch_directory+ compared_folders[0]  + folder_separator );	
		if(exit_after_analyzing_specific_data)
			exit();	
	}
	if(perform_all_missing_analyses)
	{
		print_section_separator	("Performing ROI analysis of all unprocessed reconstructed data sets", PRINT_MAJOR_SECTION);
		//reconstruction_folders_filename 			= "reconstruction_folders.txt";
		reconstruction_data_folders = file_2_array(test_batch_directory, reconstruction_folders_filename, DONT_PRINT_PATH);	
		for(i = 0; i < reconstruction_data_folders.length; i++)
		{
			current_parameter_value_test_folder	= reconstruction_data_folders[i];
			reconstruction_data_exists 			= verify_reconstruction_output(test_batch_directory, current_parameter_value_test_folder, reconstructed_image_filenames, ROI_analysis_TV_input_filename);
			missing_data 						= verify_ROI_analysis_output_files(test_batch_directory, current_parameter_value_test_folder, ROI_analysis_RSP_output_filename, ROI_analysis_RSP_error_output_filename, ROI_analysis_std_dev_output_filename, ROI_analysis_TV_input_filename, ROI_analysis_slices_2_analyze_folders, ROI_selection_diameter_folders);
			//print								("reconstruction_data_exists = " + reconstruction_data_exists);
			//print								("missing_data = " + missing_data);
			current_analysis_target 			= construct_valid_directory_path(test_batch_directory, current_parameter_value_test_folder);
			//print								("current_analysis_target = " + current_analysis_target);
			if(reconstruction_data_exists && missing_data)
				runMacro						(ROI_analysis_macro_path, current_analysis_target);
			//if(!missing_data)
			//	{print("------------>it works\n\n\n\n\n\n\n");exit();}
			//exit();	
			autobreak();
		}
		//exit();	
		//	runMacro		(ROI_analysis_macro_path, test_batch_directory+ compared_folders[0]  + folder_separator );	
		if(exit_after_performing_all_missing_analyses)
			exit();	
	}
	if(perform_multiplot_analyses)
	{
		print_section_separator	("Performing ROI analysis for all paramter value test folder reconstructions", PRINT_MAJOR_SECTION);
		//print("Analyzing compared data from:n" + compared_folders[0]);
		//specific_data_folders_filename 				= "ROI_analysis_folders.txt";
		//specific_data_folders = file_2_array(test_batch_directory, specific_data_folders_filename, DONT_PRINT_PATH);	
		//for(i = 0; i < 1; i++)
		always_perform_analyses = !only_perform_missing_analyses;
		for(i = 0; i < all_folder_strings.length; i++)
		{
			reconstructed_data_folder = all_folder_strings[i];
			print_section_separator	("Performing ROI analysis on: " + reconstructed_data_folder, PRINT_MINOR_SECTION);
			//current_parameter_value_test_folder	= reconstruction_data_folders[i];
			reconstruction_data_exists 			= verify_reconstruction_output(test_batch_directory, reconstructed_data_folder, reconstructed_image_filenames, ROI_analysis_TV_input_filename, DONT_PRINT_PATH);
			data_verified 						= verify_ROI_analysis_output_files(test_batch_directory, reconstructed_data_folder, ROI_analysis_RSP_output_filename, ROI_analysis_RSP_error_output_filename, ROI_analysis_std_dev_output_filename, ROI_analysis_TV_input_filename, ROI_analysis_slices_2_analyze_folders, ROI_selection_diameter_folders, DONT_PRINT_PATH);		
			if(reconstruction_data_exists && (data_verified || always_perform_analyses))
				runMacro		(ROI_analysis_macro_path, test_batch_directory + reconstructed_data_folder + folder_separator);
			//else
				
			autobreak();
		}
		//	runMacro		(ROI_analysis_macro_path, test_batch_directory+ compared_folders[0]  + folder_separator );	
		if(exit_after_performing_multiplot_ROI_analyses)
			exit();	
	}
//exit();	
	if(generate_specific_averaged_data)
	{
		print_section_separator	("Generating averaged data files for specific set of reconstructions", PRINT_MAJOR_SECTION);
		//averaging_folders_filename 					= "averaging_folders.txt";
		averaging_data_folders = file_2_array(test_batch_directory, averaging_folders_filename, DONT_PRINT_PATH);	
		//for(i = 0; i < 1; i++)
		for(i = 0; i < averaging_data_folders.length; i++)
		{
			//print_section_separator	("Generating averaged data files for specific set of reconstructions", PRINT_MINOR_SECTION);		
			for(ROI_selection_diameter_index = 0; ROI_selection_diameter_index < num_ROI_selection_diameters; ROI_selection_diameter_index++)
			{
				for(slice_2_analyze_index = 0; slice_2_analyze_index < num_slices_2_analyze; slice_2_analyze_index++)
				{
					slice_2_analyze						= slices_2_analyze[slice_2_analyze_index];
					slice_2_analyze_string 				= slices_2_analyze_strings[slice_2_analyze_index];
					slice_2_analyze_folder 				= slices_2_analyze_folders[slice_2_analyze_index];
					slice_2_analyze_short_string 		= slices_2_analyze_short_strings[slice_2_analyze_index];
					slice_2_analyze_short_folder 		= slices_2_analyze_short_folders[slice_2_analyze_index];
					ROI_selection_diameter				= ROI_selection_diameters[ROI_selection_diameter_index];
					ROI_selection_diameter_string 		= ROI_selection_diameter_strings[ROI_selection_diameter_index];
					ROI_selection_diameter_folder 		= ROI_selection_diameter_folders[ROI_selection_diameter_index];					
					current_analysis_subdirectory		= slice_2_analyze_folder + ROI_selection_diameter_folder;					
					current_ROI_analysis_subdirectory	= slice_2_analyze_folder + ROI_selection_diameter_folder;					
					current_TV_analysis_subdirectory	= "";					
					
					current_analysis_ROI_analysis_RSP_output_filename					= current_ROI_analysis_subdirectory 	+ folder_separator + ROI_analysis_RSP_output_filename;
					current_analysis_ROI_analysis_RSP_error_output_filename				= current_ROI_analysis_subdirectory 	+ folder_separator + ROI_analysis_RSP_error_output_filename;
					current_analysis_ROI_analysis_std_dev_output_filename				= current_ROI_analysis_subdirectory 	+ folder_separator + ROI_analysis_std_dev_output_filename;
					current_analysis_ROI_analysis_TV_output_filename					= current_TV_analysis_subdirectory 		+ folder_separator + ROI_analysis_TV_output_filename;				
				
					print								("-------> Generating average RSP data files");
					avg_RSP_data_strings 				= generate_averaged_data_CSV(test_batch_directory, averaging_data_folders[i], current_ROI_analysis_subdirectory, ROI_analysis_RSP_output_filename, iterations_2_analyze_strings, num_ROIs_2_analyze, ROIs_per_reconstruction);
					print								("-------> Generating average RSP % error data files");
					avg_RSP_error_data_strings 			= generate_averaged_data_CSV(test_batch_directory, averaging_data_folders[i], current_ROI_analysis_subdirectory, ROI_analysis_RSP_error_output_filename, iterations_2_analyze_strings, num_ROIs_2_analyze, ROIs_per_reconstruction);
					print								("-------> Generating average standard deviation data files");
					avg_std_dev_data_strings 			= generate_averaged_data_CSV(test_batch_directory, averaging_data_folders[i], current_ROI_analysis_subdirectory, ROI_analysis_std_dev_output_filename, iterations_2_analyze_strings, num_ROIs_2_analyze, ROIs_per_reconstruction);
					print								("-------> Generating average TV data files");
					avg_TV_data_strings 				= generate_averaged_data_TXT(test_batch_directory, averaging_data_folders[i], ROI_analysis_TV_input_filename, TV_step_measurements_per_reconstruction);
				}
			}
		}
		if(exit_after_averaging_specific_data)
			exit();
	}
	//exit();	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//***********************************************************************************************************************************************************************************************//
	//**************************************************** Perform comparison of analysis of individual reconstruction results **********************************************************************//
	//***********************************************************************************************************************************************************************************************//	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////							
	if(perform_test_comparisons)     
	{
		for(ROI_selection_diameter_index = 0; ROI_selection_diameter_index < num_ROI_selection_diameters; ROI_selection_diameter_index++)
		{
			for(slice_2_analyze_index = 0; slice_2_analyze_index < num_slices_2_analyze; slice_2_analyze_index++)
			{
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//***************************************************************************************************************************************************************************************//
			//************************ Establishing relevant information for the current ROI selection diameter/slice # analysis ********************************************************************//
			//***************************************************************************************************************************************************************************************//	
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////				
				current_test_start_folder_index										= 0;
				current_test_multiplot_start_folder_index							= 0;
				slice_2_analyze														= slices_2_analyze[slice_2_analyze_index];
				slice_2_analyze_string 												= slices_2_analyze_strings[slice_2_analyze_index];
				slice_2_analyze_folder 												= slices_2_analyze_folders[slice_2_analyze_index];
				slice_2_analyze_short_string 										= slices_2_analyze_short_strings[slice_2_analyze_index];
				slice_2_analyze_short_folder 										= slices_2_analyze_short_folders[slice_2_analyze_index];
				ROI_selection_diameter												= ROI_selection_diameters[ROI_selection_diameter_index];
				ROI_selection_diameter_string 										= ROI_selection_diameter_strings[ROI_selection_diameter_index];
				ROI_selection_diameter_folder 										= ROI_selection_diameter_folders[ROI_selection_diameter_index];
				
				current_analysis_subdirectory										= slice_2_analyze_folder + ROI_selection_diameter_folder;					
				current_analysis_short_subdirectory									= slice_2_analyze_short_folder + ROI_selection_diameter_folder;					
				current_ROI_analysis_subdirectory									= current_analysis_subdirectory;
				current_TV_analysis_subdirectory									= "";
				current_ROI_analysis_short_subdirectory								= current_analysis_short_subdirectory;
				current_TV_analysis_short_subdirectory								= "";
				
				current_analysis_ROI_analysis_TV_input_filename						= ROI_analysis_TV_input_filename;				
				current_analysis_ROI_analysis_RSP_output_filename					= current_ROI_analysis_subdirectory 	+ folder_separator + ROI_analysis_RSP_output_filename;
				current_analysis_ROI_analysis_RSP_error_output_filename				= current_ROI_analysis_subdirectory 	+ folder_separator + ROI_analysis_RSP_error_output_filename;
				current_analysis_ROI_analysis_std_dev_output_filename				= current_ROI_analysis_subdirectory 	+ folder_separator + ROI_analysis_std_dev_output_filename;
				current_analysis_ROI_analysis_TV_output_filename					= current_TV_analysis_subdirectory 		+ folder_separator + ROI_analysis_TV_output_filename;				
				
				current_analysis_parameter_value_test_RSP_output_filename			= current_ROI_analysis_subdirectory 	+ folder_separator + parameter_value_test_RSP_output_filename2;
				current_analysis_parameter_value_test_RSP_error_output_filename		= current_ROI_analysis_subdirectory 	+ folder_separator + parameter_value_test_RSP_error_output_filename2;
				current_analysis_parameter_value_test_std_dev_output_filename		= current_ROI_analysis_subdirectory 	+ folder_separator + parameter_value_test_std_dev_output_filename2;
				current_analysis_parameter_value_test_TV_output_filename			= current_TV_analysis_subdirectory 		+ folder_separator + parameter_value_test_TV_output_filename2;				
				//current_analysis_parameter_value_test_RSP_output_filename			= current_ROI_analysis_subdirectory 	+ folder_separator + parameter_value_test_RSP_output_filename;
				//current_analysis_parameter_value_test_RSP_error_output_filename	= current_ROI_analysis_subdirectory 	+ folder_separator + parameter_value_test_RSP_error_output_filename;
				//current_analysis_parameter_value_test_std_dev_output_filename		= current_ROI_analysis_subdirectory 	+ folder_separator + parameter_value_test_std_dev_output_filename;
				//current_analysis_parameter_value_test_TV_output_filename			= current_TV_analysis_subdirectory 		+ folder_separator + parameter_value_test_TV_output_filename;				
				
				current_analysis_multiplot_RSP_input_filename						= current_analysis_parameter_value_test_RSP_output_filename;
				current_analysis_multiplot_RSP_error_input_filename					= current_analysis_parameter_value_test_RSP_error_output_filename;
				current_analysis_multiplot_std_dev_input_filename					= current_analysis_parameter_value_test_std_dev_output_filename;
				current_analysis_multiplot_TV_input_filename						= current_analysis_parameter_value_test_TV_output_filename;				
				//exit();			
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//***************************************************************************************************************************************************************************************//
				//******** Generate plots of RSP vs. target test parameter values with individual curves for each multiplot target parameter value ***************************************//
				//***************************************************************************************************************************************************************************************//	
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																	
				all_RSP_multiplot_data 					= newArray();
				all_RSP_error_multiplot_data 			= newArray();				
				all_std_dev_multiplot_data 				= newArray();
				all_TV_multiplot_data 					= newArray();
				//*******************************************************************************************************************************************************************************//
				//************************* User defined function definitions *******************************************************************************************************************//
				//*******************************************************************************************************************************************************************************//
				//for( reduced_path_index = 0; reduced_path_index < lengthOf(all_reduced_path_strings); reduced_path_index += num_multiplot_parameter_values * num_target_test_parameter_values)
				for( i = 0; i < num_input_directories; i += num_input_directories_per_multiplot)
				{					
					current_multiplot_analysis_number										= i / num_input_directories_per_multiplot;
					print_section_separator("Performing multiplot analysis # " + current_multiplot_analysis_number + " of " + num_multiplot_analyses, true);
					current_test_start_folder 												= all_folder_strings[current_test_start_folder_index];					
					current_test_folders													= Array.slice(all_folder_strings, current_test_start_folder_index, current_test_start_folder_index + num_target_test_parameter_values);					
					current_test_folder_basename_end_index 									= lastIndexOf(current_test_start_folder, "_");
					current_test_start_folder_index											+= num_target_test_parameter_values;
					current_test_end_folder 												= all_folder_strings[current_test_start_folder_index - 1];
					current_test_folder_basename											= parameter_value_test_folder_name_2_base(current_test_start_folder, target_test_parameter_prefix);
					current_test_parameter_value_strings 									= folder_name_2_parameter_values(current_test_start_folder, parameter_string_prefixes);
					current_test_parameter_values_string									= folder_2_parameter_string(current_test_start_folder, parameter_string_prefixes, parameter_string_precisions, target_test_parameter_index - 1);					
					current_test_plot_title_parameter_values_string 						= generate_plot_title_parameter_string(current_test_parameter_value_strings, target_test_parameter_index, target_test_parameter_range_suffix, parameter_string_prefixes);
					current_test_parameter_equals_strings 									= folder_name_2_parameter_equals_strings(current_test_start_folder, parameter_string_prefixes, parameter_string_precisions, DONT_SHORTEN_STRINGS);
					current_test_parameter_equals_short_strings 							= folder_name_2_parameter_equals_strings(current_test_start_folder, parameter_string_prefixes, parameter_string_precisions, SHORTEN_STRINGS);
					current_test_file_suffix 												= generate_parameter_value_test_multiplot_copy_suffixes(current_test_parameter_value_strings, parameter_string_prefixes, num_parameter_values, parameter_string_precisions, multiplot_parameter_index, multiplot_parameter_range_filenaming);

					current_test_multiplot_start_folder 									= all_reduced_folder_strings[i];			
					current_test_multiplot_folders											= Array.slice(all_reduced_folder_strings, current_test_multiplot_start_folder_index, current_test_multiplot_start_folder_index + num_multiplot_parameter_values);					
					current_test_multiplot_start_folder_index								+= num_target_test_parameter_values * num_multiplot_parameter_values;
					current_test_multiplot_end_folder 										= all_reduced_folder_strings[current_test_multiplot_start_folder_index - 1];			
					current_test_multiplot_folder_basename										= parameter_value_test_folder_name_2_base(current_test_multiplot_start_folder, target_test_parameter_prefix);
					current_test_multiplot_value_strings 									= folder_name_2_parameter_values(current_test_multiplot_start_folder, parameter_string_prefixes);
					current_test_multiplot_value_short_strings 								= remove_trailing_zeros_set(current_test_multiplot_value_strings, parameter_string_precisions);
					
					current_test_multiplot_parameter_values_string							= folder_2_parameter_string(current_test_multiplot_start_folder, parameter_string_prefixes, parameter_string_precisions, target_test_parameter_index - 1);					
					current_test_multiplot_title_parameter_values_string 					= generate_multivalue_parameter_values_title_string(target_test_parameter_range_suffix, multiplot_parameter_index, current_test_multiplot_value_short_strings, multiplot_parameter_range_plots, parameter_string_prefixes, num_parameter_values );
					current_test_multiplot_title_other_parameter_values_string 				= generate_multivalue_parameter_values_short_title_string(target_test_parameter_index, multiplot_parameter_index, current_test_multiplot_value_short_strings, parameter_string_prefixes, num_parameter_values );
					current_test_multiplot_parameter_equals_strings 						= folder_name_2_parameter_equals_strings(current_test_multiplot_start_folder, parameter_string_prefixes, parameter_string_precisions, DONT_SHORTEN_STRINGS);
					current_test_multiplot_parameter_equals_short_strings 					= folder_name_2_parameter_equals_strings(current_test_multiplot_start_folder, parameter_string_prefixes, parameter_string_precisions, SHORTEN_STRINGS);
					current_test_multiplot_file_suffix 										= generate_multivalue_parameter_equals_suffix(current_test_multiplot_value_strings, parameter_string_short_prefixes, num_parameter_values, parameter_string_precisions, multiplot_parameter_index, multiplot_parameter_range_filenaming);
					current_test_multiplot_parameter_equals_short_string 					= current_test_multiplot_parameter_equals_short_strings[multiplot_parameter_index];
					current_test_multiplot_parameter_equals_short_string_folder 			= folder_separator + current_test_multiplot_parameter_equals_short_string;
										
					current_parameter_value_test_multiplot_parent_directory					= generate_current_parameter_value_test_multiplot_output_directory(parameter_value_tests_output_directory_parent, multiplot_parameter_index, current_test_multiplot_value_strings, multiplot_parameter_range_filenaming, target_test_parameter_range_suffix, parameter_string_prefixes, parameter_string_precisions);
					current_test_copied_ROI_analysis_suffixes								= generate_combined_filename_suffixes(multiplot_parameter_equals_short_strings, target_test_parameter_equals_short_strings);
	
					current_analysis_copied_ROI_analysis_suffixes							= target_test_parameter_equals_short_strings;
					
					/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////				
					// Output directories 
					parameter_value_tests_input_directory_parent			= test_batch_directory;
					parameter_value_tests_output_directory_parent			= test_batch_directory;	
					current_parameter_value_test_parent_directory 			= construct_valid_directory_path(test_batch_directory, parameter_value_test_identifier);
					current_parameter_value_test_plot_parent_directory 		= construct_valid_directory_path(parameter_value_tests_output_directory_parent, current_test_multiplot_folder_basename + target_test_parameter_range_suffix);
					current_parameter_value_test_multiplot_parent_directory	= generate_current_parameter_value_test_multiplot_output_directory(parameter_value_tests_output_directory_parent, multiplot_parameter_index, current_test_multiplot_value_strings, multiplot_parameter_range_filenaming, target_test_parameter_range_suffix, parameter_string_prefixes, parameter_string_precisions);
					current_parameter_value_test_parent_folder 				= File.getName(current_parameter_value_test_parent_directory);
					current_parameter_value_test_plot_parent_folder 		= File.getName(current_parameter_value_test_plot_parent_directory);
					current_parameter_value_test_multiplot_parent_folder 	= File.getName(current_parameter_value_test_multiplot_parent_directory);
					
					make_subdirectory	(parameter_value_tests_output_directory_parent, current_parameter_value_test_parent_folder,				DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
					make_subdirectory	(parameter_value_tests_output_directory_parent, current_parameter_value_test_plot_parent_folder,		DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
					make_subdirectory	(parameter_value_tests_output_directory_parent, current_parameter_value_test_multiplot_parent_folder,	DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
					
					ROI_data_subdirectories 				= newArray(num_multiplot_parameter_values);
					ROI_data_short_subdirectories 			= newArray(num_multiplot_parameter_values);
					TV_data_subdirectories 					= newArray(num_multiplot_parameter_values);
					PNG_plot_subdirectories 				= newArray(images_per_reconstruction);
					PNG_plot_short_subdirectories 			= newArray(images_per_reconstruction);
					for(j = 0; j <= num_recon_iterations; j++)
					{
						PNG_plot_subdirectories[j] 			= current_analysis_subdirectory + reconstructed_image_folders[j];
						PNG_plot_short_subdirectories[j] 	= current_analysis_short_subdirectory + reconstructed_image_folders[j];
						
					}
					for(j = 0; j < num_multiplot_parameter_values; j++)
					{
						TV_data_subdirectories[j] 			= folder_separator + multiplot_parameter_equals_short_strings[j];
						ROI_data_subdirectories[j] 			= current_analysis_subdirectory + folder_separator + multiplot_parameter_equals_short_strings[j];
						ROI_data_short_subdirectories[j] 	= current_analysis_short_subdirectory + folder_separator + multiplot_parameter_equals_short_strings[j];
					
					}
					
					current_parameter_value_test_subdirectory 										= current_test_multiplot_parameter_equals_short_string_folder;			
					current_parameter_value_test_ROI_analysis_subdirectory 							= current_analysis_subdirectory;
					current_parameter_value_test_ROI_analysis_short_subdirectory 					= current_analysis_short_subdirectory;
					current_parameter_value_test_ROI_analysis_parameter_value_subdirectory 			= construct_valid_directory_path(current_analysis_subdirectory, current_test_multiplot_parameter_equals_short_string_folder);
					current_parameter_value_test_ROI_analysis_parameter_value_short_subdirectory 	= construct_valid_directory_path(current_analysis_short_subdirectory, current_test_multiplot_parameter_equals_short_string_folder);
					current_parameter_value_test_TV_analysis_subdirectory 							= "";
					current_parameter_value_test_TV_analysis_parameter_value_subdirectory 			=  current_test_multiplot_parameter_equals_short_string_folder;
													
					current_parameter_value_test_output_directory_path 								= make_subdirectory	(current_parameter_value_test_parent_directory, multiplot_parameter_range_filenaming,				DIRECTORIES_CREATED, print_directories_created);
					current_parameter_value_test_TV_analysis_path 									= current_parameter_value_test_output_directory_path;
					current_parameter_value_test_TV_analysis_parameter_value_path 					= make_subdirectory	(current_parameter_value_test_TV_analysis_path, current_test_multiplot_parameter_equals_short_string_folder,		DIRECTORIES_CREATED, print_directories_created);
					current_parameter_value_test_ROI_analysis_path 									= make_subdirectory	(current_parameter_value_test_output_directory_path, current_parameter_value_test_ROI_analysis_short_subdirectory,		DIRECTORIES_CREATED, print_directories_created);
					current_parameter_value_test_ROI_analysis_parameter_value_path 					= make_subdirectory	(current_parameter_value_test_ROI_analysis_path, current_test_multiplot_parameter_equals_short_string_folder,	DIRECTORIES_CREATED, print_directories_created);						
					current_parameter_value_test_plot_ROI_analysis_path 							= make_subdirectory	(current_parameter_value_test_plot_parent_directory, current_analysis_subdirectory,		DIRECTORIES_CREATED, print_directories_created);
					current_parameter_value_test_plot_TV_analysis_path 								= current_parameter_value_test_plot_parent_directory;
					current_parameter_value_test_multiplot_ROI_analysis_path 						= make_subdirectory	(current_parameter_value_test_multiplot_parent_directory, current_analysis_short_subdirectory,		DIRECTORIES_CREATED, print_directories_created);
					current_parameter_value_test_multiplot_ROI_analysis_parameter_value_path 		= make_subdirectory	(current_parameter_value_test_multiplot_ROI_analysis_path, current_test_multiplot_parameter_equals_short_string_folder,		DIRECTORIES_CREATED, print_directories_created);
					current_parameter_value_test_multiplot_TV_analysis_path 						= current_parameter_value_test_multiplot_parent_directory;
					current_parameter_value_test_multiplot_TV_analysis_parameter_value_path 		= make_subdirectory	(current_parameter_value_test_multiplot_TV_analysis_path, current_test_multiplot_parameter_equals_short_string_folder,		DIRECTORIES_CREATED, print_directories_created);
									
					current_parameter_value_test_multiplot_identifier_suffix 						= current_test_multiplot_parameter_equals_short_string + "_" + target_test_parameter_range_filenaming; 
					
					parameter_value_test_multiplot_PNG_suffixes 										= newArray(ROIs_per_reconstruction);
					target_test_parameter_multiplot_PNG_suffixes 										= newArray(ROIs_per_reconstruction);
					multiplot_parameter_multiplot_PNG_suffixes 											= newArray(ROIs_per_reconstruction);								
					parameter_value_test_TV_multiplot_PNG_suffixes 										= newArray(images_per_reconstruction);
					target_test_parameter_TV_multiplot_PNG_suffixes 									= newArray(images_per_reconstruction);
					multiplot_parameter_TV_multiplot_PNG_suffixes 										= newArray(images_per_reconstruction);								
					parameter_value_test_multiplot_animation_basenames 									= newArray(num_ROIs_2_analyze);
					target_test_parameter_multiplot_animation_basenames 									= newArray(num_ROIs_2_analyze);
					multiplot_parameter_multiplot_animation_basenames 									= newArray(num_ROIs_2_analyze);								
					parameter_value_test_TV_multiplot_animation_basenames 									= newArray(TV_measurement_plots_per_iteration);
					target_test_parameter_TV_multiplot_animation_basenames 									= newArray(TV_measurement_plots_per_iteration);
					multiplot_parameter_TV_multiplot_animation_basenames 									= newArray(TV_measurement_plots_per_iteration);								
					current_parameter_value_test_ROI_analysis_iteration_paths 							= newArray(images_per_reconstruction);
					current_parameter_value_test_ROI_analysis_parameter_value_iteration_paths 			= newArray(images_per_reconstruction);
					current_parameter_value_test_plot_ROI_analysis_iteration_paths 						= newArray(images_per_reconstruction);
					current_parameter_value_test_multiplot_ROI_analysis_iteration_paths 				= newArray(images_per_reconstruction);
					current_parameter_value_test_multiplot_ROI_analysis_parameter_value_iteration_paths	= newArray(images_per_reconstruction);

					for(iteration = 0; iteration <= num_recon_iterations; iteration++)
					{
						current_parameter_value_test_ROI_analysis_iteration_path 							= make_subdirectory	(current_parameter_value_test_ROI_analysis_path, reconstructed_image_short_folders[iteration],				DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
						//current_parameter_value_test_ROI_analysis_parameter_value_iteration_path			= make_subdirectory	(current_parameter_value_test_ROI_analysis_parameter_value_path, reconstructed_image_short_folders[iteration],				DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
						current_parameter_value_test_plot_ROI_analysis_iteration_path						= make_subdirectory	(current_parameter_value_test_plot_ROI_analysis_path, reconstructed_image_folders[iteration],				DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
						current_parameter_value_test_multiplot_ROI_analysis_iteration_path					= make_subdirectory	(current_parameter_value_test_multiplot_ROI_analysis_path, reconstructed_image_short_folders[iteration],	DIRECTORIES_CREATED, PRINTING_OFF);//print_directories_created);
						//current_parameter_value_test_multiplot_ROI_analysis_parameter_value_iteration_path 	= make_subdirectory	(current_parameter_value_test_multiplot_ROI_analysis_parameter_value_path, reconstructed_image_short_folders[iteration],	DIRECTORIES_CREATED, PRINTING_OFF);	//print_directories_created);					

						current_parameter_value_test_ROI_analysis_iteration_paths[iteration] 							= current_parameter_value_test_ROI_analysis_iteration_path;
						//current_parameter_value_test_ROI_analysis_parameter_value_iteration_paths[iteration] 			= current_parameter_value_test_ROI_analysis_parameter_value_iteration_path;
						current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration] 						= current_parameter_value_test_plot_ROI_analysis_iteration_path;
						current_parameter_value_test_multiplot_ROI_analysis_iteration_paths[iteration] 					= current_parameter_value_test_multiplot_ROI_analysis_iteration_path;
						//current_parameter_value_test_multiplot_ROI_analysis_parameter_value_iteration_paths[iteration] 	= current_parameter_value_test_multiplot_ROI_analysis_parameter_value_iteration_path;
						parameter_value_test_TV_multiplot_PNG_suffixes[iteration] 	= "_" + current_test_multiplot_file_suffix + "_" + target_test_parameter_range_filenaming + "_" + reconstructed_image_short_strings[iteration] + PNG;
						target_test_parameter_TV_multiplot_PNG_suffixes[iteration] = "_" + reconstructed_image_strings[iteration] + PNG;
						multiplot_parameter_TV_multiplot_PNG_suffixes[iteration] 	= "_" + multiplot_parameter_range_filenaming + "_" + target_test_parameter_range_filenaming + "_" + reconstructed_image_short_strings[iteration] + PNG;												
						
					}
					for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
					{
						parameter_value_test_multiplot_animation_basenames[ROI] 	= "_" + current_test_multiplot_file_suffix + "_" + target_test_parameter_range_filenaming + "_" + ROI_label_nicknames[ROI] + "_" + reconstructed_image_range_string;
						target_test_parameter_multiplot_animation_basenames[ROI] = "_" + ROI_labels[ROI] + "_" + reconstructed_image_range_string;
						multiplot_parameter_multiplot_animation_basenames[ROI] 	= "_" + multiplot_parameter_range_filenaming + "_" + target_test_parameter_range_filenaming + "_" + ROI_label_nicknames[ROI] + "_" + reconstructed_image_range_string;												
						for(iteration = 0; iteration <= num_recon_iterations; iteration++)
						{
							index = ROI + iteration * num_ROIs_2_analyze;
							parameter_value_test_multiplot_PNG_suffixes[index] 	= "_" + current_test_multiplot_file_suffix + "_" + target_test_parameter_range_filenaming + "_" + ROI_label_nicknames[ROI] + "_" + reconstructed_image_short_strings[iteration] + PNG;
							target_test_parameter_multiplot_PNG_suffixes[index] = "_" + ROI_labels[ROI] + "_" + reconstructed_image_strings[iteration] + PNG;
							multiplot_parameter_multiplot_PNG_suffixes[index] 	= "_" + multiplot_parameter_range_filenaming + "_" + target_test_parameter_range_filenaming + "_" + ROI_label_nicknames[ROI] + "_" + reconstructed_image_short_strings[iteration] + PNG;												
						}											
					}
					for(TV_measurement = 0; TV_measurement < TV_measurement_plots_per_iteration; TV_measurement++)
					{
						parameter_value_test_TV_multiplot_animation_basenames[TV_measurement] 		= "_" + current_test_multiplot_file_suffix + "_" + target_test_parameter_range_filenaming + "_" + reconstructed_image_range_string;
						target_test_parameter_TV_multiplot_animation_basenames[TV_measurement] 		= "_" + reconstructed_image_range_string;
						multiplot_parameter_TV_multiplot_animation_basenames[TV_measurement] 		= "_" + multiplot_parameter_range_filenaming + "_" + target_test_parameter_range_filenaming + "_" + reconstructed_image_range_string;																				
					}
					App("\nparameter_value_test_multiplot_PNG_suffixes", parameter_value_test_multiplot_PNG_suffixes);
					App("\ntarget_test_parameter_multiplot_PNG_suffixes", target_test_parameter_multiplot_PNG_suffixes);
					App("\nmultiplot_parameter_multiplot_PNG_suffixes", multiplot_parameter_multiplot_PNG_suffixes);
					App("\nparameter_value_test_multiplot_animation_basenames", parameter_value_test_multiplot_animation_basenames);
					App("\ntarget_test_parameter_multiplot_animation_basenames", target_test_parameter_multiplot_animation_basenames);
					App("\nmultiplot_parameter_multiplot_animation_basenames", multiplot_parameter_multiplot_animation_basenames);
					//exit();
									
					
					print_current_parameter_value_test_info									();
					print_current_parameter_value_test_multiplot_info						();
					print_current_parameter_value_test_parent_directory_info				();
					//exit();				
				//*******************************************************************************************************************************************************************************//
				//************************* User defined function definitions *******************************************************************************************************************//
				//*******************************************************************************************************************************************************************************//
					RSP_multiplot_PNG_directories				= newArray();
					RSP_multiplot_PNG_filenames					= newArray();
					RSP_multiplot_PNG_paths						= newArray();
					RSP_error_multiplot_PNG_directories			= newArray();
					RSP_error_multiplot_PNG_filenames			= newArray();
					RSP_error_multiplot_PNG_paths				= newArray();
					std_dev_multiplot_PNG_directories			= newArray();
					std_dev_multiplot_PNG_filenames				= newArray();
					std_dev_multiplot_PNG_paths					= newArray();
					TV_multiplot_PNG_directories				= newArray();
					TV_multiplot_PNG_filenames					= newArray();
					TV_multiplot_PNG_paths						= newArray();
					TV_step_multiplot_PNG_directories			= newArray();
					TV_step_multiplot_PNG_filenames				= newArray();
					TV_step_multiplot_PNG_paths					= newArray();
					TV_difference_multiplot_PNG_directories		= newArray();
					TV_difference_multiplot_PNG_filenames		= newArray();
					TV_difference_multiplot_PNG_paths			= newArray();																									
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************//
//******** Import comparison data (if it exists for this test) **************************************************************************************************************************//
//***************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																	
				//compared_folders_filename 					= "compared_folders.txt";	
				compared_RSP_data_exists = false;
				compared_RSP_error_data_exists = false;
				compared_TV_data_exists = false;
				compared_std_dev_data_exists = false;
				if(add_compared_data)
				{
					multiplot_compared_data_subdirectory			= current_test_multiplot_start_folder + compared_data_folder_suffix;
					if(indexOf(multiplot_compared_data_subdirectory, "TV_0") != -1)
						multiplot_compared_data_subdirectory		= replace(multiplot_compared_data_subdirectory, "TV_0", "TV_1");//			= current_test_multiplot_start_folder + compared_data_folder_suffix;
					multiplot_compared_RSP_data_subdirectory		= construct_valid_directory_path(multiplot_compared_data_subdirectory, current_ROI_analysis_subdirectory);  
					multiplot_compared_RSP_error_data_subdirectory	= construct_valid_directory_path(multiplot_compared_data_subdirectory, current_ROI_analysis_subdirectory);  
					multiplot_compared_std_dev_data_subdirectory	= construct_valid_directory_path(multiplot_compared_data_subdirectory, current_ROI_analysis_subdirectory);  
					multiplot_compared_TV_data_subdirectory			= construct_valid_directory_path(multiplot_compared_data_subdirectory, current_TV_analysis_subdirectory);  										
					print_compared_data_subdirectory_info			();					
					print_section_separator				("Importing data used for comparison in multiplots", true);				
					if(add_compared_RSP_data)
					{
						print_section_separator	("Importing RSP data used for comparison in multiplots", false);				
						compared_RSP_data 					= import_multiplot_comparison_data(test_batch_directory, multiplot_compared_RSP_data_subdirectory, ROI_analysis_RSP_output_filename, CSV, iterations_2_analyze, num_ROIs_2_analyze);
						compared_RSP_data_exists 			= (compared_RSP_data.length > 0);			
					}
					if(add_compared_RSP_error_data)
					{
						print_section_separator	("Importing RSP error data used for comparison in multiplots", false);				
						compared_RSP_error_data 			= import_multiplot_comparison_data(test_batch_directory, multiplot_compared_RSP_error_data_subdirectory, ROI_analysis_RSP_error_output_filename, CSV, iterations_2_analyze, num_ROIs_2_analyze);
						compared_RSP_error_data_exists 		= (compared_RSP_error_data.length > 0);	
					}
					if(add_compared_std_dev_data)
					{
						print_section_separator	("Importing standard deviation data used for comparison in multiplots", false);				
						compared_std_dev_data 				= import_multiplot_comparison_data(test_batch_directory, multiplot_compared_std_dev_data_subdirectory, ROI_analysis_std_dev_output_filename, CSV, iterations_2_analyze, num_ROIs_2_analyze);
						compared_std_dev_data_exists 		= (compared_std_dev_data.length > 0);	
					}
					if(add_compared_TV_data)
					{
						print_section_separator	("Importing total variation (TV) data used for comparison in multiplots", false);				
						compared_TV_data 					= import_multiplot_comparison_data(test_batch_directory, multiplot_compared_TV_data_subdirectory, ROI_analysis_TV_input_filename, TXT, iterations_2_analyze, num_ROIs_2_analyze);
						compared_TV_data_exists 			= (compared_TV_data.length > 0);	
						if(compared_TV_data_exists)
						{
							TV_before_TVS_compared_data			= newArray(num_recon_iterations);					
							TV_after_TVS_compared_data			= newArray(num_recon_iterations);
							for(j = 0; j < num_recon_iterations; j++)
							{
								index 							= j * TV_step_measurements_per_iteration;
								TV_before_TVS_compared_data[j]	= compared_TV_data[index];
								TV_after_TVS_compared_data[j]	= compared_TV_data[index + 1];						
							}
						}		
					}				
				}
				//exit();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************//
//******** Generate plots of RSP % error vs. target test parameter values with individual curves for each multiplot target parameter value *******************************************//
//***************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																	

					if(generate_parameter_value_test_comparison_data)
					{
						print_section_separator	("Combining ROI analysis data from each reconstruction of current parameter value test and generating parameter value test comparison data files used to create multiplots", true);				
						print															("-------> Current parameter value test parameter values\n" + current_test_parameter_values_string);						
						print_comparison_data_directories_created						= false;
						parameter_test 													= i;						
						copied_ROI_analysis_RSP_multiplot_output_subdirectories			= subdirectory_array_concatenation(multiplot_parameter_equals_short_strings, current_ROI_analysis_subdirectory, "");
						copied_ROI_analysis_RSP_error_multiplot_output_subdirectories	= subdirectory_array_concatenation(multiplot_parameter_equals_short_strings, current_ROI_analysis_subdirectory, "");
						copied_ROI_analysis_std_dev_multiplot_output_subdirectories		= subdirectory_array_concatenation(multiplot_parameter_equals_short_strings, current_ROI_analysis_subdirectory, "");
						copied_ROI_analysis_TV_multiplot_output_subdirectories			= subdirectory_array_concatenation(multiplot_parameter_equals_short_strings, current_TV_analysis_subdirectory, "");					
						if(generate_parameter_value_test_RSP_comparison_data)
						{
							print_section_separator	("Combining RSP data from ROI analysis of each reconstruction of current parameter value test", false);				
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								RSP_comparison_data 						= newArray();						
								multiplot_folder 							= all_reduced_path_strings					[i + multiplot];
								current_parameter_values 					= folder_name_2_parameter_values 			(multiplot_folder, parameter_string_prefixes);
								parameter_string 							= folder_2_multiplot_string					(multiplot_folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								multiplot_folder_basename					= parameter_value_test_folder_name_2_base	(multiplot_folder, parameter_string_prefixes[target_test_parameter_index]);
								copied_ROI_analysis_RSP_output_directory	= multiplot_folder_basename + target_test_parameter_range_suffix;
								copied_ROI_analysis_RSP_output_path			= construct_valid_directory_path	(copied_ROI_analysis_RSP_output_directory, current_ROI_analysis_subdirectory);
								for(N = 0; N < num_target_test_parameter_values; N++)
								{
									copied_ROI_analysis_RSP_multiplot_output_subdirectory	= copied_ROI_analysis_RSP_multiplot_output_subdirectories[multiplot];
									copied_ROI_analysis_RSP_multiplot_output_directory		= construct_valid_directory_path(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_RSP_multiplot_output_subdirectory);
									folder						= multiplot_folder_basename + "_" + target_test_parameter_value_strings[N];
									data 						= csv_2_array(folder, current_analysis_ROI_analysis_RSP_output_filename, iterations_2_analyze_strings, num_ROIs_2_analyze, print_input_CSV_path);
									RSP_comparison_data 		= Array.concat(RSP_comparison_data, data);																							
									make_subdirectory			(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_RSP_multiplot_output_subdirectory, DIRECTORIES_CREATED, print_comparison_data_directories_created);					
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_RSP_output_filename, copied_ROI_analysis_RSP_output_path, RSP_data_file_basenames, current_analysis_copied_ROI_analysis_suffixes[N], CSV, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);	
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_RSP_output_filename, copied_ROI_analysis_RSP_multiplot_output_directory, RSP_data_file_basenames, current_test_copied_ROI_analysis_suffixes[multiplot * num_target_test_parameter_values + N], CSV, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);										
								}  
								parameter_value_test_RSP_output_directory				= construct_valid_directory_path(multiplot_folder_basename + target_test_parameter_range_suffix, current_analysis_subdirectory);
								parameter_value_test_RSP_output_filename				= parameter_value_test_RSP_output_filename2;
								copied_parameter_value_test_RSP_output_directory 		= generate_current_parameter_value_test_multiplot_output_directory(test_batch_directory, multiplot_parameter_index, current_test_multiplot_value_strings, multiplot_parameter_range_filenaming, target_test_parameter_range_suffix, parameter_string_prefixes, parameter_string_precisions);
								copied_parameter_value_test_RSP_output_filename 		= RSP_data_file_basenames + "_" + multiplot_parameter_equals_short_strings[multiplot] + "_" + target_test_parameter_range_filenaming + TXT;
								copied_parameter_value_test_RSP_output_path 			= construct_valid_directory_path(copied_parameter_value_test_RSP_output_directory, current_ROI_analysis_subdirectory);
								if(write_RSP_comparison_data)			
								{	
									array_2_TXT			(parameter_value_test_RSP_output_directory, parameter_value_test_RSP_output_filename, RSP_comparison_data, print_output_TXT_path, overwrite_RSP_comparison_data);			
									array_2_TXT			(copied_parameter_value_test_RSP_output_path, copied_parameter_value_test_RSP_output_filename, RSP_comparison_data, print_output_TXT_path, overwrite_RSP_comparison_data);			
								}
							}						
						}
						if(generate_parameter_value_test_RSP_error_comparison_data)
						{
							print_section_separator	("Combining RSP error data from ROI analysis of each reconstruction of current parameter value test", false);				
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								RSP_error_comparison_data 	= newArray();						
								multiplot_folder 			= all_reduced_path_strings					[i + multiplot];
								current_parameter_values 	= folder_name_2_parameter_values 			(multiplot_folder, parameter_string_prefixes);
								parameter_string 			= folder_2_multiplot_string					(multiplot_folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								multiplot_folder_basename	= parameter_value_test_folder_name_2_base	(multiplot_folder, parameter_string_prefixes[target_test_parameter_index]);
								copied_ROI_analysis_RSP_error_output_directory	= multiplot_folder_basename + target_test_parameter_range_suffix;
								copied_ROI_analysis_RSP_error_output_path	= construct_valid_directory_path	(copied_ROI_analysis_RSP_error_output_directory, current_ROI_analysis_subdirectory);
								for(N = 0; N < num_target_test_parameter_values; N++)
								{
									folder						= multiplot_folder_basename + "_" + target_test_parameter_value_strings[N];
									data 						= csv_2_array(folder, current_analysis_ROI_analysis_RSP_error_output_filename, iterations_2_analyze_strings, num_ROIs_2_analyze, print_input_CSV_path);
									RSP_error_comparison_data 	= Array.concat(RSP_error_comparison_data, data);																							
									copied_ROI_analysis_RSP_error_multiplot_output_subdirectory	= copied_ROI_analysis_RSP_error_multiplot_output_subdirectories[multiplot];
									copied_ROI_analysis_RSP_error_multiplot_output_directory	= construct_valid_directory_path(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_RSP_error_multiplot_output_subdirectory);
									make_subdirectory(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_RSP_error_multiplot_output_subdirectory, DIRECTORIES_CREATED, print_comparison_data_directories_created);					
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_RSP_error_output_filename, copied_ROI_analysis_RSP_error_output_path, RSP_error_data_file_basenames, current_analysis_copied_ROI_analysis_suffixes[N], CSV, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);	
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_RSP_error_output_filename, copied_ROI_analysis_RSP_error_multiplot_output_directory, RSP_error_data_file_basenames, current_test_copied_ROI_analysis_suffixes[multiplot * num_target_test_parameter_values + N], CSV, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);										
								}  
								parameter_value_test_RSP_error_output_directory				= construct_valid_directory_path(multiplot_folder_basename + target_test_parameter_range_suffix, current_analysis_subdirectory);
								parameter_value_test_RSP_error_output_filename				= parameter_value_test_RSP_error_output_filename2;
								copied_parameter_value_test_RSP_error_output_directory 		= generate_current_parameter_value_test_multiplot_output_directory(test_batch_directory, multiplot_parameter_index, current_test_multiplot_value_strings, multiplot_parameter_range_filenaming, target_test_parameter_range_suffix, parameter_string_prefixes, parameter_string_precisions);
								copied_parameter_value_test_RSP_error_output_filename 		= RSP_error_data_file_basenames + "_" + multiplot_parameter_equals_short_strings[multiplot] + "_" + target_test_parameter_range_filenaming + TXT;
								copied_parameter_value_test_RSP_error_output_path 			= construct_valid_directory_path(copied_parameter_value_test_RSP_error_output_directory, current_ROI_analysis_subdirectory);						
								if(write_RSP_error_comparison_data)			
								{	
									array_2_TXT			(parameter_value_test_RSP_error_output_directory, parameter_value_test_RSP_error_output_filename, RSP_error_comparison_data, print_output_TXT_path, overwrite_RSP_error_comparison_data);			
									array_2_TXT			(copied_parameter_value_test_RSP_error_output_path, copied_parameter_value_test_RSP_error_output_filename, RSP_error_comparison_data, print_output_TXT_path, overwrite_RSP_error_comparison_data);			
								}
							}	
						}
						if(generate_parameter_value_test_std_dev_comparison_data)
						{
							print_section_separator	("Combining RSP standard deviation data from ROI analysis of each reconstruction of current parameter value test", false);				
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								std_dev_comparison_data 	= newArray();						
								multiplot_folder 			= all_reduced_path_strings					[i + multiplot];
								current_parameter_values 	= folder_name_2_parameter_values 			(multiplot_folder, parameter_string_prefixes);
								parameter_string 			= folder_2_multiplot_string					(multiplot_folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								multiplot_folder_basename	= parameter_value_test_folder_name_2_base	(multiplot_folder, parameter_string_prefixes[target_test_parameter_index]);
								copied_ROI_analysis_std_dev_output_directory	= multiplot_folder_basename + target_test_parameter_range_suffix;
								copied_ROI_analysis_std_dev_output_path	= construct_valid_directory_path	(copied_ROI_analysis_std_dev_output_directory, current_ROI_analysis_subdirectory);
								for(N = 0; N < num_target_test_parameter_values; N++)
								{
									folder						= multiplot_folder_basename + "_" + target_test_parameter_value_strings[N];
									data 						= csv_2_array(folder, current_analysis_ROI_analysis_std_dev_output_filename, iterations_2_analyze_strings, num_ROIs_2_analyze, print_input_CSV_path);
									std_dev_comparison_data 	= Array.concat(std_dev_comparison_data, data);																							
									copied_ROI_analysis_std_dev_multiplot_output_subdirectory	= copied_ROI_analysis_std_dev_multiplot_output_subdirectories[multiplot];
									copied_ROI_analysis_std_dev_multiplot_output_directory		= construct_valid_directory_path(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_std_dev_multiplot_output_subdirectory);
									make_subdirectory(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_std_dev_multiplot_output_subdirectory, DIRECTORIES_CREATED, print_comparison_data_directories_created);					
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_std_dev_output_filename, copied_ROI_analysis_std_dev_output_path, std_dev_data_file_basenames, current_analysis_copied_ROI_analysis_suffixes[N], CSV, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);	
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_std_dev_output_filename, copied_ROI_analysis_std_dev_multiplot_output_directory, std_dev_data_file_basenames, current_test_copied_ROI_analysis_suffixes[multiplot * num_target_test_parameter_values + N], CSV, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);										
								}  
								parameter_value_test_std_dev_output_directory				= construct_valid_directory_path(multiplot_folder_basename + target_test_parameter_range_suffix, current_analysis_subdirectory);
								parameter_value_test_std_dev_output_filename				= parameter_value_test_std_dev_output_filename2;
								copied_parameter_value_test_std_dev_output_directory 		= generate_current_parameter_value_test_multiplot_output_directory(test_batch_directory, multiplot_parameter_index, current_test_multiplot_value_strings, multiplot_parameter_range_filenaming, target_test_parameter_range_suffix, parameter_string_prefixes, parameter_string_precisions);
								copied_parameter_value_test_std_dev_output_filename 		= std_dev_data_file_basenames + "_" + multiplot_parameter_equals_short_strings[multiplot] + "_" + target_test_parameter_range_filenaming + TXT;
								copied_parameter_value_test_std_dev_output_path 			= construct_valid_directory_path(copied_parameter_value_test_std_dev_output_directory, current_ROI_analysis_subdirectory);
								
								if(write_std_dev_comparison_data)			
								{	
									array_2_TXT			(parameter_value_test_std_dev_output_directory, parameter_value_test_std_dev_output_filename, std_dev_comparison_data, print_output_TXT_path, overwrite_std_dev_comparison_data);			
									array_2_TXT			(copied_parameter_value_test_std_dev_output_path, copied_parameter_value_test_std_dev_output_filename, std_dev_comparison_data, print_output_TXT_path, overwrite_std_dev_comparison_data);			
								}
							}	
						}
						if(generate_parameter_value_test_TV_comparison_data)
						{
							print_section_separator	("Combining TV data from ROI analysis of each reconstruction of current parameter value test", false);				
							TV_measurements_per_reconstruction 	= 2 * num_recon_iterations;		
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								TVS_comparison_data 			= newArray();						
								multiplot_folder 			= all_reduced_path_strings					[i + multiplot];
								current_parameter_values 	= folder_name_2_parameter_values 			(multiplot_folder, parameter_string_prefixes);
								parameter_string 			= folder_2_multiplot_string					(multiplot_folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								multiplot_folder_basename	= parameter_value_test_folder_name_2_base	(multiplot_folder, parameter_string_prefixes[target_test_parameter_index]);
								copied_ROI_analysis_TV_output_directory	= multiplot_folder_basename + target_test_parameter_range_suffix;
								copied_ROI_analysis_TV_output_path	= construct_valid_directory_path	(copied_ROI_analysis_TV_output_directory, current_TV_analysis_subdirectory);
								for(N = 0; N < num_target_test_parameter_values; N++)
								{
									folder						= multiplot_folder_basename + "_" + target_test_parameter_value_strings[N];
									data 						= file_2_array(folder, current_analysis_ROI_analysis_TV_input_filename, print_input_TXT_path);
									TVS_comparison_data 		= Array.concat(TVS_comparison_data, data);																							
									copied_ROI_analysis_TV_multiplot_output_subdirectory	= copied_ROI_analysis_TV_multiplot_output_subdirectories[multiplot];
									copied_ROI_analysis_TV_multiplot_output_directory		= construct_valid_directory_path(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_TV_multiplot_output_subdirectory);
									make_subdirectory			(current_parameter_value_test_multiplot_parent_directory, copied_ROI_analysis_TV_multiplot_output_subdirectory, DIRECTORIES_CREATED, print_comparison_data_directories_created);					
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_TV_input_filename, copied_ROI_analysis_TV_output_path, TV_data_file_basenames, current_analysis_copied_ROI_analysis_suffixes[N], TXT, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);	
									copy_input_data_file_to 	= copy_input_data_file	(folder, current_analysis_ROI_analysis_TV_input_filename, copied_ROI_analysis_TV_multiplot_output_directory, TV_data_file_basenames, current_test_copied_ROI_analysis_suffixes[multiplot * num_target_test_parameter_values + N], TXT, COPIED_FILE_FROM_LIST, COPIED_FILE_TO_LIST, print_copied_file_paths);																	
									COPIED_FILE_TO_LIST			= Array.concat(COPIED_FILE_TO_LIST, copy_input_data_file_to);										
								}  
								parameter_value_test_TV_output_directory			= multiplot_folder_basename + target_test_parameter_range_suffix;
								parameter_value_test_TV_output_filename				= parameter_value_test_TV_output_filename2;
								copied_parameter_value_test_TV_output_directory 	= generate_current_parameter_value_test_multiplot_output_directory(test_batch_directory, multiplot_parameter_index, current_test_multiplot_value_strings, multiplot_parameter_range_filenaming, target_test_parameter_range_suffix, parameter_string_prefixes, parameter_string_precisions);
								copied_parameter_value_test_TV_output_filename 		= TV_data_file_basenames + "_" + target_test_parameter_range_filenaming + "_" + multiplot_parameter_equals_short_strings[multiplot] + TXT;
								copied_parameter_value_test_TV_output_path 			= construct_valid_directory_path(copied_parameter_value_test_TV_output_directory, current_TV_analysis_subdirectory);
								
								if(write_TV_comparison_data)			
								{	
									array_2_TXT			(parameter_value_test_TV_output_directory, parameter_value_test_TV_output_filename, TVS_comparison_data, print_output_TXT_path, overwrite_TV_comparison_data);			
									array_2_TXT			(copied_parameter_value_test_TV_output_path, copied_parameter_value_test_TV_output_filename, TVS_comparison_data, print_output_TXT_path, overwrite_TV_comparison_data);			
								}
							}	
							//exit();
						}			
						//Array.print(COPIED_FILE_TO_LIST);
						print("=======> Finished generating parameter value test comparison data files");				
						//exit();	
					}
					//exit();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************//
//******** Generate plots of RSP % error vs. target test parameter values with individual curves for each multiplot target parameter value *******************************************//
//***************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																											
						//generate_RSP_multiplots = false;
						if(generate_RSP_multiplots)
						{		
							print_section_separator		("Generating RSP multiplots", true);
							if(compared_RSP_data_exists)
								print									("-------> RSP multiplots will include comparison data");
							if(write_RSP_multiplots)
								print									("-------> RSP multiplots will be written to disk as PNG");
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								folder 								= all_reduced_folder_strings		[i + multiplot];
								current_parameter_values 			= folder_name_2_parameter_values 	(folder, parameter_string_prefixes);
								parameter_string 					= folder_2_multiplot_string			(folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								current_parameter_test_directory 	= test_batch_directory + substring	(folder, 0, lengthOf(folder) - lengthOf(current_parameter_values[target_test_parameter_index]) - 1) + target_test_parameter_range_suffix;
								RSP 								= file_2_array						(current_parameter_test_directory, current_analysis_multiplot_RSP_input_filename, print_input_data_path);	
								reordered_RSP_multiplot_data 		= order_comparison_data				(RSP, indices_4_ordering_data, num_ROIs_2_analyze, images_per_reconstruction, num_target_test_parameter_values);
								all_RSP_multiplot_data 				= Array.concat						(all_RSP_multiplot_data, reordered_RSP_multiplot_data);
							}
							for(iteration = 1; iteration <= num_recon_iterations; iteration++)
							{
								current_iteration_RSP_multiplot_directory 	= current_parameter_value_test_plot_ROI_analysis_path + reconstructed_image_folders[iteration];
								directory_created_successfully				= make_directory_recorded(current_iteration_RSP_multiplot_directory, DIRECTORIES_CREATED, print_directories_created);
								for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
								{							
									start_multiplot_data_index		= (iteration - 1) * ROIs_per_test_iteration + ROI * num_target_test_parameter_values;		
									all_current_multiplot_data		= newArray();
									for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 				= start_multiplot_data_index + multiplot * ROIs_per_test;
										partial_multiplot_data 		= Array.slice(all_RSP_multiplot_data, start_index, start_index + num_target_test_parameter_values);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, partial_multiplot_data);					
									}								
									primary_multiplot_data 			= Array.slice(all_current_multiplot_data, 0, num_target_test_parameter_values);								
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									title 							= "RSP vs. Nk in " + ROI_labels[ROI] + " : iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_other_parameter_values_string;
									text 							= "RSP vs. Nk in " + ROI_labels[ROI] + " : iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_other_parameter_values_string;
									common_legend_text				= parameter_string_prefixes[multiplot_parameter_index] + " = ";
									legend_entries 					= generate_legend_entries(common_legend_text, multiplot_parameter_values);						
									xlabel 							= "Nk [TVS repetitions]";
									ylabel 							= "RSP";
									Plot.create						( title, xlabel, ylabel, target_test_parameter_values, primary_multiplot_data 									);
									Plot.setLimits					( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax								);
									color_rotation_index 			= 0;
									set_plot_props					(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, color_rotation[color_rotation_index++], linewidth	);
									for(multiplot = 1; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 				= multiplot * num_target_test_parameter_values;
										add_multiplot_data 			= Array.slice(all_current_multiplot_data, start_index, start_index + num_target_test_parameter_values);
										Plot.setColor				(color_rotation[color_rotation_index++]																					);	
										Plot.add					( "line", target_test_parameter_values, add_multiplot_data														);																																													
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, add_multiplot_data);	
									}	
									if(add_compared_RSP_data && compared_RSP_data_exists)
									{									
										index 						= ROI + (iteration-1) * num_ROIs_2_analyze;
										compared_RSP_line 			= newArray(target_test_parameter_values.length);
										Array.fill					(compared_RSP_line, compared_RSP_data[index]);
										Plot.setColor				(color_rotation[color_rotation_index++]);									
										Plot.add					("line", target_test_parameter_values, compared_RSP_line);	
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, compared_RSP_line);										
										legend_entries 				= Array.concat(legend_entries, "OTVS");
										set_plot_props				(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, add_line_color, linewidth);																		
									}
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									Plot.setLimits					( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax								);
									Plot.setLimitsToFit				();	
									set_plot_data_props				(color_rotation[0], axis_label_text_size, x_frame_size, y_frame_size									);									
									//run							("Size...", "width=" + x_frame_size + " height=" + y_frame_size + " constrain average interpolation=Bilinear");	
									//run							("Canvas Size...", "width=612" + " height=510" + " position=Center");
									if(write_RSP_multiplots)
									{								
										index 						= ROI + iteration * num_ROIs_2_analyze;
										plot_filepath 				= save_PNG(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		RSP_output_basenames + target_test_parameter_multiplot_PNG_suffixes[index], 	overwrite_RSP_multiplots, CLOSE_PNG, 	DONT_PRINT_PATH);														
										RSP_multiplot_PNG_paths 	= Array.concat(RSP_multiplot_PNG_paths, plot_filepath);										
										copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		RSP_output_basenames 		+ target_test_parameter_multiplot_PNG_suffixes[index], current_parameter_value_test_ROI_analysis_iteration_paths[iteration], 			RSP_output_short_basenames 	+ parameter_value_test_multiplot_PNG_suffixes[index], DONT_PRINT_PATH);
										copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		RSP_output_basenames 		+ target_test_parameter_multiplot_PNG_suffixes[index], current_parameter_value_test_multiplot_ROI_analysis_iteration_paths[iteration], RSP_output_short_basenames 	+ multiplot_parameter_multiplot_PNG_suffixes[index], DONT_PRINT_PATH);
									
									}																	
									//exit();					
								}// END for( n = 0; n < num_ROIs_2_analyze; n++)								
							}// END for(iteration = 1; iteration <= num_recon_iterations; iteration++)	
							//exit();
							//***********************************************************************************************************************************************************************//
							//************************* User defined function definitions ***********************************************************************************************************//
							//***********************************************************************************************************************************************************************//							
							RSP_multiplot_PNG_multiplots_exist 				= verify_distributed_files(RSP_multiplot_PNG_paths);
							if(!RSP_multiplot_PNG_multiplots_exist)
								print										("-------> One or more of the RSP multiplot PNGs to be combined into animations are missing"); 		
							if(generate_RSP_multiplot_animations && RSP_multiplot_PNG_multiplots_exist)
							{
								print_section_separator			("Creating RSP multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs", false);
								//print										("RSP_multiplot_PNG_paths:"); 		
								//Array.print(RSP_multiplot_PNG_paths);
								for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
								{
									stack_title 					= RSP_data_file_basenames + "_" + ROI_labels[ROI] + "_" + reconstructed_image_range_string;
									PNG_subsequence_paths 			= PNG_sequence_2_stack(RSP_multiplot_PNG_paths, ROI, num_ROIs_2_analyze, num_recon_iterations, RSP_Booleans, RSP_data_file_basenames, PNG_image_Booleans, stack_title, animated_GIF_info, AVI_info, true);
									stack_2_animated_GIF			(stack_title, current_parameter_value_test_plot_ROI_analysis_path, 		RSP_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, GIF_frame_rate, write_RSP_multiplot_animated_GIFs, overwrite_RSP_multiplot_animated_GIFs, PRINT_PATH, CLOSE_WINDOW);
									copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_path, 		RSP_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, current_parameter_value_test_ROI_analysis_path, 			RSP_output_short_basenames 	+ parameter_value_test_multiplot_animation_basenames[ROI] 	+ GIF, DONT_PRINT_PATH);
									copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_path, 		RSP_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, current_parameter_value_test_multiplot_ROI_analysis_path, RSP_output_short_basenames 	+ multiplot_parameter_multiplot_animation_basenames[ROI] 	+ GIF, DONT_PRINT_PATH);							
								}
								//exit();
							}
							//exit();								
						}// END if(generate_RSP_multiplots)	
						//exit();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************//
//******** Generate plots of RSP % error vs. target test parameter values with individual curves for each multiplot target parameter value *******************************************//
//***************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																	
						//generate_RSP_error_multiplots = false;
						if(generate_RSP_error_multiplots)
						{				
							print_section_separator			("Generating RSP % Error multiplots", true);
							if(compared_RSP_error_data_exists)
								print									("-------> RSP % error multiplots will include comparison data");
							if(write_RSP_error_multiplots)
								print									("-------> RSP % error multiplots will be written to disk as PNG");
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								folder 									= all_reduced_folder_strings[i + multiplot];
								current_parameter_values 				= folder_name_2_parameter_values 	(folder, parameter_string_prefixes);
								parameter_string 						= folder_2_multiplot_string			(folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								current_parameter_test_directory 		= test_batch_directory + substring	(folder, 0, lengthOf(folder) -lengthOf(current_parameter_values[target_test_parameter_index]) - 1)+ target_test_parameter_range_suffix;
								RSP_error 								= file_2_array						(current_parameter_test_directory, current_analysis_multiplot_RSP_error_input_filename, print_input_data_path);	
								reordered_RSP_error_multiplot_data 		= order_comparison_data				(RSP_error, indices_4_ordering_data, num_ROIs_2_analyze, images_per_reconstruction, num_target_test_parameter_values);
								all_RSP_error_multiplot_data 			= Array.concat						(all_RSP_error_multiplot_data, reordered_RSP_error_multiplot_data);
							}
							function values_2_absolute_values(_array)
									{
										for(j = 0; j < _array.length; j++)						
											_array[j] = abs(_array[j]);
										return _array;
									}
									all_RSP_error_multiplot_data = values_2_absolute_values(all_RSP_error_multiplot_data);	
							
							for(iteration = 1; iteration <= num_recon_iterations; iteration++)
							{
								current_iteration_RSP_error_multiplot_directory = current_parameter_value_test_multiplot_parent_directory + current_ROI_analysis_subdirectory + folder_separator + reconstructed_image_folders[iteration];
								directory_created_successfully					= make_directory_recorded(current_iteration_RSP_error_multiplot_directory, DIRECTORIES_CREATED, print_directories_created);
								for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
								{							
									start_multiplot_data_index		= (iteration - 1) * ROIs_per_test_iteration + ROI * num_target_test_parameter_values;		
									all_current_multiplot_data		= newArray();
									for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 				= start_multiplot_data_index + multiplot * ROIs_per_test;
										partial_multiplot_data 		= Array.slice(all_RSP_error_multiplot_data, start_index, start_index + num_target_test_parameter_values);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, partial_multiplot_data);					
									}		
									
									primary_multiplot_data 			= Array.slice(all_current_multiplot_data, 0, num_target_test_parameter_values);								
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									title 							= "RSP % Error vs. Nk in " + ROI_labels[ROI] + " : iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_other_parameter_values_string;
									text 							= "RSP % Error vs. Nk in " + ROI_labels[ROI] + " : iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_other_parameter_values_string;
									common_legend_text				= parameter_string_prefixes[multiplot_parameter_index] + " = ";
									legend_entries 					= generate_legend_entries(common_legend_text, multiplot_parameter_values);						
									xlabel 							= "Nk [TVS repetitions]";
									ylabel 							= "RSP % Error";
									Plot.create						( title, xlabel, ylabel, target_test_parameter_values, primary_multiplot_data 							);
									Plot.setLimits					( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax							);
									color_rotation_index 			= 0;
									set_plot_props					(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, color_rotation[color_rotation_index++], linewidth	);
									for(multiplot = 1; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 				= multiplot * num_target_test_parameter_values;
										add_multiplot_data 			= Array.slice(all_current_multiplot_data, start_index, start_index + num_target_test_parameter_values);
										add_multiplot_data 			= values_2_absolute_values(add_multiplot_data);										
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, add_multiplot_data);										
										Plot.setColor				(color_rotation[color_rotation_index++]);	
										Plot.add					( "line", target_test_parameter_values, add_multiplot_data		);																																													
									}
									if(add_compared_RSP_error_data && compared_RSP_error_data_exists)
									{									
										index 						= ROI + (iteration-1) * num_ROIs_2_analyze;
										compared_RSP_error_line 	= newArray(target_test_parameter_values.length);
										Array.fill					(compared_RSP_error_line, compared_RSP_error_data[index]);
										compared_RSP_error_line 	= values_2_absolute_values(compared_RSP_error_line);	
										Plot.setColor				(color_rotation[color_rotation_index++]);									
										Plot.add					("line", target_test_parameter_values, compared_RSP_error_line);	
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, compared_RSP_error_line);										
										legend_entries 				= Array.concat(legend_entries, "OTVS");
										set_plot_props				(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, color_rotation[color_rotation_index++], linewidth	);
									}
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									Plot.setLimits					( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax								);
									set_plot_data_props				(color_rotation[0], axis_label_text_size, x_frame_size, y_frame_size									);									
									if(write_RSP_error_multiplots)
									{
										index 						= ROI + iteration * num_ROIs_2_analyze;
										plot_filepath 				= save_PNG(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		RSP_error_output_basenames + target_test_parameter_multiplot_PNG_suffixes[index], 	overwrite_RSP_error_multiplots, CLOSE_PNG, 	DONT_PRINT_PATH);														
										RSP_error_multiplot_PNG_paths 	= Array.concat(RSP_error_multiplot_PNG_paths, plot_filepath);										
										copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		RSP_error_output_basenames 		+ target_test_parameter_multiplot_PNG_suffixes[index], current_parameter_value_test_ROI_analysis_iteration_paths[iteration], 			RSP_error_output_short_basenames 	+ parameter_value_test_multiplot_PNG_suffixes[index], DONT_PRINT_PATH);
										copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		RSP_error_output_basenames 		+ target_test_parameter_multiplot_PNG_suffixes[index], current_parameter_value_test_multiplot_ROI_analysis_iteration_paths[iteration], RSP_error_output_short_basenames 	+ multiplot_parameter_multiplot_PNG_suffixes[index], DONT_PRINT_PATH);
									
									}
									//close();									
								}// END for( n = 0; n < num_ROIs_2_analyze; n++)								
							}// END for(iteration = 1; iteration <= num_recon_iterations; iteration++)		
							//exit();					
							//***********************************************************************************************************************************************************************//
							//************************* User defined function definitions ***********************************************************************************************************//
							//***********************************************************************************************************************************************************************//
							RSP_error_multiplot_PNG_multiplots_exist 				= verify_distributed_files(RSP_error_multiplot_PNG_paths);
							if(!RSP_error_multiplot_PNG_multiplots_exist)
								print										("-------> One of more of the RSP error multiplot PNGs to be combined into animations are missing"); 		
							if(generate_RSP_error_multiplot_animations && RSP_error_multiplot_PNG_multiplots_exist)
							{
								print_section_separator		("Creating RSP % error multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs", false);
								//print									("RSP_error_multiplot_PNG_paths:"); 	Array.print(RSP_error_multiplot_PNG_paths);
								for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
								{
									stack_title 					= RSP_error_data_file_basenames + "_" + ROI_labels[ROI] + "_" + reconstructed_image_range_string;
									PNG_subsequence_paths 			= PNG_sequence_2_stack(RSP_error_multiplot_PNG_paths, ROI, num_ROIs_2_analyze, num_recon_iterations, RSP_error_Booleans, RSP_error_data_file_basenames, PNG_image_Booleans, stack_title, animated_GIF_info, AVI_info, true);
									stack_2_animated_GIF			(stack_title, current_parameter_value_test_plot_ROI_analysis_path, 		RSP_error_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, GIF_frame_rate, write_RSP_error_multiplot_animated_GIFs, overwrite_RSP_error_multiplot_animated_GIFs, PRINT_PATH, CLOSE_WINDOW);
									copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_path, 		RSP_error_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, current_parameter_value_test_ROI_analysis_path, 			RSP_error_output_short_basenames 	+ parameter_value_test_multiplot_animation_basenames[ROI] 	+ GIF, DONT_PRINT_PATH);
									copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_path, 		RSP_error_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, current_parameter_value_test_multiplot_ROI_analysis_path, RSP_error_output_short_basenames 	+ multiplot_parameter_multiplot_animation_basenames[ROI] 	+ GIF, DONT_PRINT_PATH);
									//exit();
								}
							}																				
						}// END if(generate_RSP_error_multiplots)			
						//exit();					
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************//
//******** Generate plots of Standard Deviation vs. target test parameter values with individual curves for each multiplot target parameter value *******************************************//
//***************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																	
						//generate_std_dev_multiplots = false;
						if(generate_std_dev_multiplots)
						{				
							print_section_separator		("Generating standard deviation multiplots", true);
							if(compared_std_dev_data_exists)
								print								("-------> RSP standard deviation multiplots will include comparison data");
							if(write_std_dev_multiplots)
								print									("-------> RSP standard deviation multiplots will be written to disk as PNG");
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								folder 								= all_reduced_folder_strings[i + multiplot];
								current_parameter_values 			= folder_name_2_parameter_values (folder, parameter_string_prefixes);
								parameter_string 					= folder_2_multiplot_string	(folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								current_parameter_test_directory 	= test_batch_directory + substring(folder, 0, lengthOf(folder) -lengthOf(current_parameter_values[target_test_parameter_index]) - 1)+ target_test_parameter_range_suffix;
								std_dev 							= file_2_array				(current_parameter_test_directory, current_analysis_multiplot_std_dev_input_filename, print_input_data_path);	
								reordered_std_dev_multiplot_data 	= order_comparison_data		(std_dev, indices_4_ordering_data, num_ROIs_2_analyze, images_per_reconstruction, num_target_test_parameter_values);
								all_std_dev_multiplot_data 			= Array.concat				(all_std_dev_multiplot_data, reordered_std_dev_multiplot_data);
							}		
							for(iteration = 1; iteration <= num_recon_iterations; iteration++)
							{
								current_iteration_std_dev_multiplot_directory 	= current_parameter_value_test_multiplot_parent_directory + current_ROI_analysis_subdirectory + folder_separator + reconstructed_image_folders[iteration];
								directory_created_successfully					= make_directory_recorded(current_iteration_std_dev_multiplot_directory, DIRECTORIES_CREATED, print_directories_created);
								for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
								{							
									start_multiplot_data_index		= (iteration - 1) * ROIs_per_test_iteration + ROI * num_target_test_parameter_values;		
									all_current_multiplot_data		= newArray();
									for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 				= start_multiplot_data_index + multiplot * ROIs_per_test;
										partial_multiplot_data 		= Array.slice(all_std_dev_multiplot_data, start_index, start_index + num_target_test_parameter_values);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, partial_multiplot_data);					
									}								
									primary_multiplot_data 			= Array.slice(all_current_multiplot_data, 0, num_target_test_parameter_values);								
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									title 							= "Standard Deviation vs. Nk in " + ROI_labels[ROI] + " : iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_other_parameter_values_string;
									text 							= "Standard Deviation vs. Nk in " + ROI_labels[ROI] + " : iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_other_parameter_values_string;
									common_legend_text				= parameter_string_prefixes[multiplot_parameter_index] + " = ";
									legend_entries 					= generate_legend_entries(common_legend_text, multiplot_parameter_values);						
									xlabel 							= "Nk [TVS repetitions]";
									ylabel 							= "Standard Deviation";
									Plot.create						( title, xlabel, ylabel, target_test_parameter_values, primary_multiplot_data 							);
									Plot.setLimits					( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax							);
									color_rotation_index 			= 0;
									set_plot_props					(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, color_rotation[color_rotation_index++], linewidth	);
									for(multiplot = 1; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 				= multiplot * num_target_test_parameter_values;
										add_multiplot_data 			= Array.slice(all_current_multiplot_data, start_index, start_index + num_target_test_parameter_values);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, add_multiplot_data);										
										Plot.setColor				(color_rotation[color_rotation_index++]);									
										Plot.add					( "line", target_test_parameter_values, add_multiplot_data		);																																													
									}
									if(add_compared_std_dev_data && compared_std_dev_data_exists)
									{									
										index 						= ROI + (iteration-1) * num_ROIs_2_analyze;
										compared_std_dev_line 		= newArray(target_test_parameter_values.length);
										Array.fill					(compared_std_dev_line, compared_std_dev_data[index]);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, compared_std_dev_line);										
										Plot.setColor				(color_rotation[color_rotation_index++]);									
										Plot.add					("line", target_test_parameter_values, compared_std_dev_line);	
										legend_entries 				= Array.concat(legend_entries, "OTVS");
										set_plot_props				(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, add_line_color, linewidth);																		
									}	
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									Plot.setLimits					( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax								);
									set_plot_data_props				(color_rotation[0], axis_label_text_size, x_frame_size, y_frame_size									);									
									if(write_std_dev_multiplots)
									{
										index 						= ROI + iteration * num_ROIs_2_analyze;
										plot_filepath 				= save_PNG(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		std_dev_output_basenames + target_test_parameter_multiplot_PNG_suffixes[index], 	overwrite_std_dev_multiplots, CLOSE_PNG, 	DONT_PRINT_PATH);														
										std_dev_multiplot_PNG_paths 	= Array.concat(std_dev_multiplot_PNG_paths, plot_filepath);										
										copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		std_dev_output_basenames 		+ target_test_parameter_multiplot_PNG_suffixes[index], current_parameter_value_test_ROI_analysis_iteration_paths[iteration], 			std_dev_output_short_basenames 	+ parameter_value_test_multiplot_PNG_suffixes[index], DONT_PRINT_PATH);
										copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_iteration_paths[iteration], 		std_dev_output_basenames 		+ target_test_parameter_multiplot_PNG_suffixes[index], current_parameter_value_test_multiplot_ROI_analysis_iteration_paths[iteration], std_dev_output_short_basenames 	+ multiplot_parameter_multiplot_PNG_suffixes[index], DONT_PRINT_PATH);									
									}
									//close();									
								}// END for( n = 0; n < num_ROIs_2_analyze; n++)								
							}// END for(iteration = 1; iteration <= num_recon_iterations; iteration++)	
							//***********************************************************************************************************************************************************************//
							//************************* User defined function definitions ***********************************************************************************************************//
							//***********************************************************************************************************************************************************************//
							std_dev_multiplot_PNG_multiplots_exist 				= verify_distributed_files(std_dev_multiplot_PNG_paths);
							if(!std_dev_multiplot_PNG_multiplots_exist)
								print										("-------> One of more of the RSP standard deviation multiplot PNGs to be combined into animations are missing"); 		
							if(generate_std_dev_multiplot_animations && std_dev_multiplot_PNG_multiplots_exist)
							{
								print_section_separator		("Creating RSP standard deviation multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs", false);
								//print									("std_dev_multiplot_PNG_paths:"); 	Array.print(std_dev_multiplot_PNG_paths);
								for(ROI = 0; ROI < num_ROIs_2_analyze; ROI++)
								{
									stack_title 					= std_dev_data_file_basenames + "_" + ROI_labels[ROI] + "_" + reconstructed_image_range_string;
									PNG_subsequence_paths 			= PNG_sequence_2_stack(std_dev_multiplot_PNG_paths, ROI, num_ROIs_2_analyze, num_recon_iterations, std_dev_Booleans, std_dev_data_file_basenames, PNG_image_Booleans, stack_title, animated_GIF_info, AVI_info, true);
									stack_2_animated_GIF			(stack_title, current_parameter_value_test_plot_ROI_analysis_path, 		std_dev_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, GIF_frame_rate, write_std_dev_multiplot_animated_GIFs, overwrite_std_dev_multiplot_animated_GIFs, PRINT_PATH, CLOSE_WINDOW);
									copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_path, 		std_dev_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, current_parameter_value_test_ROI_analysis_path, 			std_dev_output_short_basenames 	+ parameter_value_test_multiplot_animation_basenames[ROI] 	+ GIF, DONT_PRINT_PATH);
									copy_distributed_file(current_parameter_value_test_plot_ROI_analysis_path, 		std_dev_output_basenames 		+ target_test_parameter_multiplot_animation_basenames[ROI] 	+ GIF, current_parameter_value_test_multiplot_ROI_analysis_path, std_dev_output_short_basenames 	+ multiplot_parameter_multiplot_animation_basenames[ROI] 	+ GIF, DONT_PRINT_PATH);
									
								}
							}														
						}// END if(generate_std_dev_multiplots)			
						//exit();					
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************//
//******** Generate plots of TV measurement vs. target test parameter values with individual curves for each multiplot target parameter value *******************************************//
//***************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////																	
						if(generate_TV_multiplots)
						{
							print_section_separator		("Generating total variation (TV) multiplots and writing them to disk as PNG", true);
							if(compared_TV_data_exists)
								print								("-------> TV multiplots will include comparison data");
							if(plot_TV_multiplots)
								print								("-------> TV before/after TVS multiplots will be written to disk as PNG");
							if(plot_TV_step_multiplots)
								print								("-------> TV step multiplots will be written to disk as PNG");
							if(plot_TV_difference_multiplots)
								print								("-------> TV difference before/after TVS multiplots will be written to disk as PNG");
										
							for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
							{
								folder 								= all_reduced_folder_strings[i + multiplot];
								current_parameter_values 			= folder_name_2_parameter_values (folder, parameter_string_prefixes);
								parameter_string 					= folder_2_multiplot_string	(folder, parameter_string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions);
								current_parameter_test_directory 	= test_batch_directory + substring(folder, 0, lengthOf(folder) - lengthOf(current_parameter_values[target_test_parameter_index]) - 1) + target_test_parameter_range_suffix;
								TV_data 							= file_2_array				(current_parameter_test_directory, current_analysis_multiplot_TV_input_filename, print_input_data_path);	
								all_TV_multiplot_data 				= Array.concat				(all_TV_multiplot_data, TV_data);
							}
							num_source_index_values 					= newArray(TV_step_measurements_per_iteration, num_recon_iterations, num_target_test_parameter_values, num_multiplot_parameter_values);
							num_destination_index_values 				= newArray(TV_step_measurements_per_iteration, num_target_test_parameter_values, num_multiplot_parameter_values);					
							for(iteration = 1; iteration <= num_recon_iterations; iteration++)
							{
								TV_step_multiplot_data									= newArray(TV_step_measurements_per_multiplot);			
								before_TVS_multiplot_data								= newArray(TV_measurements_per_multiplot);			
								after_TVS_multiplot_data								= newArray(TV_measurements_per_multiplot);			
								TV_difference_multiplot_data							= newArray(TV_difference_measurements_per_multiplot);			
								current_iteration_TV_multiplot_directory 				= current_parameter_value_test_multiplot_parent_directory;
								current_iteration_TV_step_multiplot_directory 			= current_parameter_value_test_multiplot_parent_directory;
								current_iteration_TV_difference_multiplot_directory 	= current_parameter_value_test_multiplot_parent_directory;
								directory_created_successfully							= make_directory_recorded(current_iteration_TV_multiplot_directory, DIRECTORIES_CREATED, print_directories_created);
								directory_created_successfully							= make_directory_recorded(current_iteration_TV_step_multiplot_directory, DIRECTORIES_CREATED, print_directories_created);
								directory_created_successfully							= make_directory_recorded(current_iteration_TV_difference_multiplot_directory, DIRECTORIES_CREATED, print_directories_created);									
								for(multiplot = 0; multiplot < num_multiplot_parameter_values; multiplot++)
								{
									for(N = 0; N < num_target_test_parameter_values; N++)
									{
										source_indices 											= newArray(before_TVS_index, iteration - 1, N, multiplot);
										destination_indices 									= newArray(before_TVS_index, N, multiplot);
										source_index 											= parameter_indices_2_index(num_source_index_values, source_indices);
										TV_step_destination_index 								= parameter_indices_2_index(num_destination_index_values, destination_indices);
										TV_destination_index									= TV_step_destination_index / TV_step_measurements_per_iteration;
										before_TVS_multiplot_data_value							= parseFloat(all_TV_multiplot_data[source_index]);
										after_TVS_multiplot_data_value							= parseFloat(all_TV_multiplot_data[source_index + 1]);
										before_TVS_multiplot_data[TV_destination_index]			= before_TVS_multiplot_data_value;
										after_TVS_multiplot_data[TV_destination_index]			= after_TVS_multiplot_data_value;
										TV_difference_multiplot_data[TV_destination_index]		= after_TVS_multiplot_data_value - before_TVS_multiplot_data_value;								
										TV_difference_multiplot_data[TV_destination_index]		= parseFloat(after_TVS_multiplot_data[TV_destination_index]) - parseFloat(before_TVS_multiplot_data[TV_destination_index]);
										TV_step_multiplot_data[TV_step_destination_index] 		= all_TV_multiplot_data[source_index];
										TV_step_multiplot_data[TV_step_destination_index + 1] 	= all_TV_multiplot_data[source_index + 1];																										
									}
								}		
								//exit();					
								//***********************************************************************************************************************************************************************//
								//************************* User defined function definitions ***********************************************************************************************************//
								//***********************************************************************************************************************************************************************//
								if(plot_TV_multiplots)
								{
									print						("------->Generating multiplots of total variation (TV) after TVS and writing them to disk as PNG...");
									x_axis_data					= target_test_parameter_values;
									all_current_multiplot_data 	= after_TVS_multiplot_data;										
									primary_TV_multiplot_data 	= Array.slice(after_TVS_multiplot_data, 0, TV_measurements_per_multiplot_curve);														
									plot_extrema 				= set_plot_extrema(after_TVS_multiplot_data, primary_TV_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 						= plot_extrema[0];
									ymax 						= plot_extrema[1];
									title 							= "Total Variation vs. Nk after "  + iterations_2_analyze_strings[iteration] + " iterations" + "\n" + current_test_multiplot_title_other_parameter_values_string;
									text 							= "Total Variation vs. Nk after " + iterations_2_analyze_strings[iteration] + " iterations" + "\n" + current_test_multiplot_title_other_parameter_values_string;
									common_TV_legend_text		=  parameter_string_prefixes[multiplot_parameter_index] + " = ";
									legend_entries 				= generate_legend_entries(common_TV_legend_text, multiplot_parameter_values											);						
									xlabel 						= "Nk [TVS repetitions]";
									ylabel 						= "Total Variation (TV)";
									Plot.create					( title, xlabel, ylabel, x_axis_data, primary_TV_multiplot_data 													);
									Plot.setLimits				( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax									);
									color_rotation_index 		= 0;
									set_plot_props				(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, color_rotation[color_rotation_index++], linewidth	);
									for(multiplot = 1; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 			= multiplot * TV_measurements_per_multiplot_curve;
										add_TV_multiplot_data 	= Array.slice(after_TVS_multiplot_data, start_index, start_index + TV_measurements_per_multiplot_curve				);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, add_TV_multiplot_data);										
										Plot.setColor			(color_rotation[color_rotation_index++]																					);	
										Plot.add				( "line", x_axis_data, add_TV_multiplot_data																		);																																													
									}
									if(add_compared_TV_data && compared_TV_data_exists)
									{									
										compared_TV_line 		= newArray(x_axis_data.length);
										index					= 2*(iteration - 1) + 1;
										Array.fill				(compared_TV_line, compared_TV_data[index]);
										all_current_multiplot_data 	= Array.concat(all_current_multiplot_data, compared_TV_line);										
										Plot.setColor			(color_rotation[color_rotation_index++]);									
										Plot.add				("line", x_axis_data, compared_TV_line);
										legend_entries 			= Array.concat(legend_entries, "OTVS");
										set_plot_props			(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_text_justification, add_line_color, linewidth);														
									}
									plot_extrema 					= set_plot_extrema(all_current_multiplot_data, primary_TV_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 							= plot_extrema[0];
									ymax 							= plot_extrema[1];
									Plot.setLimits					(target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax								);
									set_plot_data_props				(color_rotation[0], axis_label_text_size, x_frame_size, y_frame_size									);									
									if(write_TV_multiplots)
									{
										plot_filepath 				= save_PNG(current_parameter_value_test_plot_TV_analysis_path, 		TV_output_basenames + target_test_parameter_TV_multiplot_PNG_suffixes[iteration], 	overwrite_TV_multiplots, CLOSE_PNG, 	DONT_PRINT_PATH);														
										TV_multiplot_PNG_paths 	= Array.concat(TV_multiplot_PNG_paths, plot_filepath);										
										path  = copy_distributed_file(current_parameter_value_test_plot_TV_analysis_path, 		TV_output_basenames 		+ target_test_parameter_TV_multiplot_PNG_suffixes[iteration], current_parameter_value_test_TV_analysis_path, 			TV_output_short_basenames 	+ parameter_value_test_TV_multiplot_PNG_suffixes[iteration], DONT_PRINT_PATH);
										multiplot_path = copy_distributed_file(current_parameter_value_test_plot_TV_analysis_path, 		TV_output_basenames 		+ target_test_parameter_TV_multiplot_PNG_suffixes[iteration], current_parameter_value_test_multiplot_TV_analysis_path, TV_output_short_basenames 	+ multiplot_parameter_TV_multiplot_PNG_suffixes[iteration], DONT_PRINT_PATH);
									}								
								}// END if(plot_TV_multiplots)
								//***********************************************************************************************************************************************************************//
								//************************* User defined function definitions ***********************************************************************************************************//
								//***********************************************************************************************************************************************************************//					
								plot_TV_step_multiplots = false;
								if(plot_TV_step_multiplots)
								{
									print								("-------> Generating multiplots of total variation (TV) before/after TVS and writing them to disk as PNG...");
									x_axis_data							= newArray(TV_step_measurements_per_multiplot_curve);
									x_axis_data_min						= target_test_parameter_values[0];
									for(j = 0; j < num_target_test_parameter_values; j++)
									{
										x_axis_data[2*j] 				= target_test_parameter_values[j];				
										x_axis_data[2*j + 1] 			= target_test_parameter_values[j];
									}
									primary_TV_step_multiplot_data 		= Array.slice(TV_step_multiplot_data, 0, TV_step_measurements_per_multiplot_curve);														
									plot_extrema 						= set_plot_extrema(TV_step_multiplot_data, primary_TV_step_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale	);
									ymin 								= plot_extrema[0];
									ymax 								= plot_extrema[1];
									title 								= "TV vs. # of TVS repetitions (Nk) after " + iterations_2_analyze_strings[iteration] + " iterations";
									text 								= "Effect of TV check on total variation (TV) for iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_parameter_values_string;
									common_TV_legend_text				= "TV vs. Nk for " + parameter_string_prefixes[multiplot_parameter_index] + " = ";
									legend_entries 						= generate_legend_entries(common_TV_legend_text, multiplot_parameter_values																		);						
									xlabel 								= "Nk [TVS repetitions]";
									ylabel 								= "Total Variation (TV)";
									Plot.create							( title, xlabel, ylabel, x_axis_data, primary_TV_step_multiplot_data 																			);
									Plot.setLimits						( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax																);
									color_rotation_index 				= 0;
									set_plot_props						(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, color_rotation[color_rotation_index++], linewidth	);													
									for(multiplot = 1; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 					= multiplot * TV_step_measurements_per_multiplot_curve;
										add_TV_step_multiplot_data 		= Array.slice(TV_step_multiplot_data, start_index, start_index + TV_step_measurements_per_multiplot_curve										);
										Plot.setColor					(color_rotation[color_rotation_index++]																													);	
										Plot.add						( "line", x_axis_data, add_TV_step_multiplot_data																								);																																													
									}
									if(add_comparison_lines_2_plots && compared_TV_data_exists)
									{									
										compared_TV_line 				= newArray(x_axis_data.length);
										Array.fill						(compared_TV_line, compared_TV_data[2*(iteration - 1) + 1]);
										Plot.setColor					(color_rotation[color_rotation_index++]);									
										Plot.add						("line", x_axis_data, compared_TV_line);
										legend_entries 					= Array.concat(legend_entries, "Old TVS");
										set_plot_props					(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, add_line_color, linewidth);														
									}
									set_plot_data_props					(color_rotation[0], add_text_justification																										);								
									if(write_TV_step_multiplots)
									{
										index 							= iteration * num_ROIs_2_analyze;//RSP_data_file_base_nicknames, RSP_data_file_basenames, RSP_error_data_file_base_nicknames
										save_PNG						(current_parameter_value_test_TV_analysis_path, 			TV_step_output_short_basenames + parameter_value_test_TV_multiplot_PNG_suffixes[iteration], 	overwrite_TV_step_multiplots, DONT_CLOSE_PNG, 	print_output_PNG_paths);														
										save_PNG						(current_parameter_value_test_multiplot_TV_analysis_path, 	TV_step_output_short_basenames + multiplot_parameter_TV_multiplot_PNG_suffixes[iteration], overwrite_TV_step_multiplots, CLOSE_PNG, 		print_output_PNG_paths);																																					
									}							
								}// END if(plot_TV_step_multiplots)		
								//***********************************************************************************************************************************************************************//
								//************************* User defined function definitions ***********************************************************************************************************//
								//***********************************************************************************************************************************************************************//					
								if(plot_TV_difference_multiplots)						
								{
									x_axis_data					= target_test_parameter_values;
									primary_multiplot_data 		= Array.slice(TV_difference_multiplot_data, 0, TV_measurements_per_multiplot_curve);														
									plot_extrema 				= set_plot_extrema(primary_multiplot_data, primary_multiplot_data, tolerance, lower_limits_scale, upper_limits_scale, difference_scale);
									ymin 						= plot_extrema[0];
									ymax 						= plot_extrema[1];
									title 						= "Difference in TV before/after TVS vs. # of TVS repetitions (Nk) after " + iterations_2_analyze_strings[iteration] + " iterations";
									text 						= "Effect of TV check on total variation (TV) difference before/after TVS for iteration " + iterations_2_analyze_strings[iteration] + "\n" + current_test_multiplot_title_parameter_values_string;
									common_TV_legend_text		= "TV vs. Nk for " + parameter_string_prefixes[multiplot_parameter_index] + " = ";
									legend_entries 				= generate_legend_entries(common_TV_legend_text, multiplot_parameter_values											);						
									xlabel 						= "Nk [TVS repetitions]";
									ylabel 						= "Total Variation (TV)";
									Plot.create					( title, xlabel, ylabel, x_axis_data, primary_multiplot_data 													);
									Plot.setLimits				( target_test_parameter_min_value, target_test_parameter_max_value, ymin, ymax									);
									color_rotation_index 		= 0;
									set_plot_props				(legend_entries, legend_text_size, x_frame_size, y_frame_size, text, text_size, text_xpos_ratio, text_ypos_ratio, color_rotation[color_rotation_index++], linewidth	);
									for(multiplot = 1; multiplot < num_multiplot_parameter_values; multiplot++)
									{
										start_index 			= multiplot * TV_measurements_per_multiplot_curve;
										add_multiplot_data 		= Array.slice(multiplot_data, start_index, start_index + TV_measurements_per_multiplot_curve				);
										Plot.setColor			(color_rotation[color_rotation_index++]																						);	
										Plot.add				( "line", x_axis_data, add_multiplot_data																		);																																													
									}
									set_plot_data_props			( color_rotation[0], add_text_justification																			);								
									if(write_TV_difference_multiplots)
									{
										index 							= iteration * num_ROIs_2_analyze;//RSP_data_file_base_nicknames, RSP_data_file_basenames, RSP_error_data_file_base_nicknames
										save_PNG						(current_parameter_value_test_TV_analysis_path, 			TV_difference_output_short_basenames + parameter_value_test_TV_multiplot_PNG_suffixes[iteration], 	overwrite_TV_difference_multiplots, DONT_CLOSE_PNG, 	print_output_PNG_paths);														
										save_PNG						(current_parameter_value_test_multiplot_TV_analysis_path, 	TV_difference_output_short_basenames + multiplot_parameter_TV_multiplot_PNG_suffixes[iteration], overwrite_TV_difference_multiplots, CLOSE_PNG, 		print_output_PNG_paths);																																					
									}				
								}// END if(plot_TV_step_multiplots)		
							}// END for(iteration = 1; iteration <= num_recon_iterations; iteration++)
							//exit();
							//***********************************************************************************************************************************************************************//
							//************************* User defined function definitions ***********************************************************************************************************//
							//***********************************************************************************************************************************************************************//
							if(generate_TV_animations)
							{
								print_section_separator				("Creating TV multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs", false);
								TV_multiplot_PNGs_exist 				= verify_distributed_files(TV_multiplot_PNG_paths);
								TV_step_multiplot_PNGs_exist 			= verify_distributed_files(TV_step_multiplot_PNG_paths);
								TV_difference_multiplot_PNGs_exist 	= verify_distributed_files(TV_difference_multiplot_PNG_paths);
								if(!TV_multiplot_PNGs_exist)
									print										("-------> One or more of the TV multiplot PNGs to be combined into animations are missing"); 		
								if(!TV_step_multiplot_PNGs_exist)
									print										("-------> One or more of the TV step multiplot PNGs to be combined into animations are missing"); 		
								if(!TV_difference_multiplot_PNGs_exist)
									print										("-------> One or more of the TV difference multiplot PNGs to be combined into animations are missing"); 								
								//exit();
								if(generate_TV_multiplot_animations && TV_multiplot_PNGs_exist)
								{
									print						("------->Creating TV multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs...");
									//print						("TV_multiplot_PNG_paths:"); 	Array.print(TV_multiplot_PNG_paths);
									//print						("TV_multiplot_PNG_paths elements" + TV_multiplot_PNG_paths.length); 	
									for(TV_measurement = 0; TV_measurement < TV_measurement_plots_per_iteration; TV_measurement++)
									{
										stack_title 					= TV_data_file_basenames + "_" + reconstructed_image_range_string;		
										PNG_subsequence_paths 			= PNG_sequence_2_stack(TV_multiplot_PNG_paths, TV_measurement, TV_measurement_plots_per_iteration, num_recon_iterations, TV_Booleans, TV_data_file_basenames, PNG_image_Booleans, stack_title, animated_GIF_info, AVI_info, true);
										stack_2_animated_GIF			(stack_title, current_parameter_value_test_plot_TV_analysis_path, 		TV_output_basenames 		+ target_test_parameter_TV_multiplot_animation_basenames[TV_measurement] 	+ GIF, GIF_frame_rate, write_TV_multiplot_animated_GIFs, overwrite_TV_multiplot_animated_GIFs, PRINT_PATH, CLOSE_WINDOW);
										copy_distributed_file(current_parameter_value_test_plot_TV_analysis_path, 		TV_output_basenames 		+ target_test_parameter_TV_multiplot_animation_basenames[TV_measurement] 	+ GIF, current_parameter_value_test_TV_analysis_path, 			TV_output_short_basenames 	+ parameter_value_test_TV_multiplot_animation_basenames[TV_measurement] + GIF, PRINT_PATH);
										copy_distributed_file(current_parameter_value_test_plot_TV_analysis_path, 		TV_output_basenames 		+ target_test_parameter_TV_multiplot_animation_basenames[TV_measurement] 	+ GIF, current_parameter_value_test_multiplot_TV_analysis_path, TV_output_short_basenames 	+ multiplot_parameter_TV_multiplot_animation_basenames[TV_measurement] 	+ GIF, PRINT_PATH);										
									}
								}
								//close();
								//exit();
								generate_TV_step_multiplot_animations = false;
								if(generate_TV_step_multiplot_animations && TV_step_multiplot_PNGs_exist)
								{
									print						("------->Creating TV step multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs...");
									//print						("TV_step_multiplot_PNG_paths:"); 	Array.print(TV_step_multiplot_PNG_paths);
									//print						("TV_step_multiplot_PNG_paths elements" + TV_step_multiplot_PNG_paths.length); 	
									for(TV_step_measurement = 0; TV_step_measurement < TV_measurement_plots_per_iteration; TV_step_measurement++)
									{																	
										animation_filename 		= TV_step_data_file_basenames + "_" + reconstructed_image_range_string;
										PNG_subsequence_paths 	= PNG_sequence_2_animations(current_parameter_value_test_multiplot_parent_directory, TV_step_multiplot_PNG_paths, TV_step_measurement, TV_measurement_plots_per_iteration, num_recon_iterations, TV_step_Booleans, TV_step_data_file_basenames, PNG_image_Booleans, animation_filename, animated_GIF_info, AVI_info, true);
										copied_animations_suffix		= current_test_multiplot_file_suffix + "_" + reconstructed_image_range_string;
										//print					("PNG_subsequence_paths:"); Array.print(PNG_subsequence_paths);
										//print					("animation_filename = " + animation_filename); 							
									}		
								}
								//close();
								//exit();
								generate_TV_difference_multiplot_animations = false;
								if(generate_TV_difference_multiplot_animations && TV_difference_multiplot_PNGs_exist)
								{
									print						("------->Creating TV difference multiplot stacks for each ROI as a function of iteration and writing stacks to animated GIFs and/or AVIs...");
									//print						("TV_difference_multiplot_PNG_paths:"); 	Array.print(TV_difference_multiplot_PNG_paths);
									//print						("TV_difference_multiplot_PNG_paths elements" + TV_difference_multiplot_PNG_paths.length); 	
									for(TV_difference_measurement = 0; TV_difference_measurement < TV_measurement_plots_per_iteration; TV_difference_measurement++)
									{
										animation_filename 		= TV_difference_data_file_basenames + "_" + reconstructed_image_range_string;
										PNG_subsequence_paths 	= PNG_sequence_2_animations(current_parameter_value_test_multiplot_parent_directory, TV_difference_multiplot_PNG_paths, TV_measurement, TV_measurement_plots_per_iteration, num_recon_iterations, TV_difference_Booleans, TV_difference_data_file_basenames, PNG_image_Booleans, animation_filename, animated_GIF_info, AVI_info, true);
										copied_animations_suffix		= current_test_multiplot_file_suffix + "_" + reconstructed_image_range_string;
										//print					("PNG_subsequence_paths:"); Array.print(PNG_subsequence_paths);
										//print					("animation_filename = " + animation_filename); 							
									}
								}		
							}
							//close();
							//exit();											
						}// END if(generate_TV_multiplots)	
						print_section_separator("FINISHED CURRENT MULTIPLOT ANALYSIS", true);
					}// END for( i = 0; i < lengthOf(all_reduced_path_strings); i+=num_input_directories_per_multiplot)											
				//exit();			
			}// END for(slice_2_analyze_index = 0; slice_2_analyze_index < num_slices_2_analyze; slice_2_analyze_index++)		
		}// END for(ROI_selection_diameter_index = 0; ROI_selection_diameter_index < num_ROI_selection_diameters; ROI_selection_diameter_index++)
	}// END if(perform_test_comparisons)
	//close_all_windows(true, false);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************************//
//************************* Directories/files created ***********************************************************************************************************************************************//
//***************************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	print_section_separator("Input/output directories and filenames created during execution", true);
	print		("-------> INPUT_FILE_LIST:\n"			); 	Array.print(INPUT_FILE_LIST); 						
	print		("-------> OUTPUT_FILE_LIST:\n"			); 	Array.print(OUTPUT_FILE_LIST); 						
	//print		("-------> COPIED_FILE_LIST:\n"			); 	Array.print(COPIED_FILE_LIST); 						
	print		("-------> COPIED_FILE_FROM_LIST:\n"	); 	Array.print(COPIED_FILE_FROM_LIST); 				
	print		("-------> COPIED_FILE_TO_LIST:\n"		); 	Array.print(COPIED_FILE_TO_LIST); 					
	print		("-------> DIRECTORIES_CREATED:\n"		); 	Array.print(DIRECTORIES_CREATED); 						
}// END macro multiplotting
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************************************************************************************************************************************************************************//
//************************* User defined function definitions ***************************************************************************************************************************************//
//***************************************************************************************************************************************************************************************************//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function Ap(arg)
{
	Array.print(arg);
}
function App(statement, array)
{
	print("*******> " + statement + " = ");
	Array.print(array);
}
function array_2_CSV(_rows, _columns, _array, _directory, _filename, _print_path, _overwrite_existing_CSV)
{
	for(_row = 0; _row < _rows; _row++)
		for (_column = 0; _column < _columns; _column++)
			setResult		(_column, _row, _array[_row * _columns + _column]);	
	_CSV_filename 			= construct_valid_file_path(_directory, _filename);		
	if(_print_path)
		print				("------->Writing to CSV at\n" + _CSV_filename);
	if(!File.exists(_CSV_filename) || _overwrite_existing_CSV)						
	{
		saveAs				("Measurements", _CSV_filename);			
		OUTPUT_FILE_LIST	= Array.concat(OUTPUT_FILE_LIST, _CSV_filename);					
	}
	run						("Clear Results");
}	
function array_2_string(array, precisions)
{
	string 			= "";
	for(i = 0; i < lengthOf(array); i++)
	{
		if( precisions[i] == -1 )
			string 	+= array[i] + " ";
		else
			string 	+= d2s(array[i], precisions[i]) + " ";
	}
	return string;
}
function array_2_TXT(path, filename, data_array, print_path, overwrite_existing_TXT)	
{
	TXT_filename 			= construct_valid_file_path(path, filename);		
	if(print_path)
		print				("------->Writing to TXT at:\n" + TXT_filename);
	if(!File.exists(TXT_filename) || overwrite_existing_TXT)
	{
		filehandle 			= File.open(TXT_filename);		
		for (i = 0; i < data_array.length; i++)
			print			(filehandle, data_array[i]);
		File.close			(filehandle);	
		OUTPUT_FILE_LIST	= Array.concat(OUTPUT_FILE_LIST, TXT_filename);
	}					
}
function autobreak()
{
	autobreak_boolean 	= file_2_array(ImageJ_macro_directory, auto_break_filename, DONT_PRINT_PATH);
	if(autobreak_boolean[0])
		exit();
}
function calculate_TV(path, ROWS, COLUMNS, SLICES)
{
	run									("Text Image... ", "open="+ path );
	run									("Image to Results");
	TV 									= 0.0;
	INPUT_FILE_LIST						= Array.concat(INPUT_FILE_LIST, path);
	for( slice = 0; slice < SLICES; slice++ )
	{
		for( row = 0; row < ROWS - 1; row++ )
		{
			for( column = 0; column < COLUMNS - 1; column++ )
			{
				table_row 				= slice * ROWS + row;
				table_row_string 		= "Y" + d2s	(table_row, 0);
				table_col_string 		= "X" + d2s	(column, 0);
				table_col_right_string 	= "X" + d2s	(column + 1, 0);
				target_voxel_value 		= getResult	(table_col_string, table_row);
				right_voxel_value 		= getResult	(table_col_right_string, table_row);
				bottom_voxel_value 		= getResult	(table_col_string , table_row + 1);				
				col_diff 				= right_voxel_value - target_voxel_value;
				row_diff 				= bottom_voxel_value - target_voxel_value;
				TV 						+= sqrt		(col_diff * col_diff + row_diff * row_diff);
			}
		}
	}
	return TV;
}
function calculate_TV_opened_image( ROWS, COLUMNS, SLICES )
{
	run									("Image to Results");
	TV 									= 0.0;
	for( slice = 0; slice < SLICES; slice++ )
	{
		for( row = 0; row < ROWS - 1; row++ )
		{
			for( column = 0; column < COLUMNS - 1; column++ )
			{
				table_row 				= slice * ROWS + row;
				table_row_string 		= "Y" + d2s(table_row, 0);
				table_col_string 		= "X" + d2s(column, 0);
				table_col_right_string 	= "X" + d2s(column + 1, 0);
				target_voxel_value 		= getResult(table_col_string, table_row);
				right_voxel_value 		= getResult(table_col_right_string, table_row);
				bottom_voxel_value 		= getResult(table_col_string , table_row + 1);				
				col_diff 				= right_voxel_value - target_voxel_value;
				row_diff 				= bottom_voxel_value - target_voxel_value;
				TV 						+= sqrt(col_diff * col_diff + row_diff * row_diff);
			}
		}
	}
	return TV;
}
function ceil(value)
{
	if(value - round(value) > 0)
		return round(value + 1);
	else
		return round(value);
}
function close_all_windows(clear_results, recording_on)
{
	run					("Close All");
	if(clear_results)
		run				("Clear Results");
	if(recording_on)
		run				("Record...");
	list 				= getList("window.titles");
	for(i = 0; i < list.length; i++ )
	{
		selectWindow	(list[i]);
		run				("Close");		
	}
}
function close_windows(clear_results, close_all, recording_on)
{
	if(close_all)
		run				("Close All");
	else
		run				("Close");
	if(clear_results)
		run				("Clear Results");
	if(recording_on)
		run				("Record...");
	list 				= getList("window.titles");
	for(i = 0; i < list.length; i++ )
	{
		selectWindow	(list[i]);
		run				("Close");		
	}
}
function construct_image_set_filenames(common_filename_prefix, image_set_filename_endings, image_extension)
{
	num_images					= image_set_filename_endings.length;
	image_set_filenames 		= newArray(num_images);
	for(i = 0; i < num_images; i++)
		image_set_filenames[i] 	= common_filename_prefix + "_" + image_set_filename_endings[i] + image_extension;
		return image_set_filenames;
}
function construct_valid_directory_path(directory, subdirectory)
{
	while(startsWith(subdirectory, File.separator))
		subdirectory = substring(subdirectory, 1, lengthOf(subdirectory));
	while(endsWith(directory, File.separator))
		directory = substring(directory, 0, lengthOf(directory) - 1);
	//path 			= directory + File.separator + subdirectory;
	return 			directory + File.separator + subdirectory;
}
function construct_valid_file_path(directory, filename)
{
	while(startsWith(filename, File.separator))
		filename = substring(filename, 1, lengthOf(filename));
	while(endsWith(directory, File.separator))
		directory = substring(directory, 0, lengthOf(directory) - 1);
	//path 			= directory + File.separator + filename;
	return			directory + File.separator + filename;
}
function copy_distributed_file(_from_directory, _from_filename, _to_directory, _to_filename, _print_to_path)
{
	_from_path = construct_valid_file_path(_from_directory, _from_filename);
	_to_path = construct_valid_file_path(_to_directory, _to_filename);
	File.copy(_from_path, _to_path);
	if(_print_to_path)
		print("File copied to :\n" + _to_path);
	return _to_path;
}
function copy_input_data_file(_input_directory, _input_filename, _output_directory, _output_basename, _output_suffix, _file_extension, _copied_from_log, _copied_to_log, _print_paths)
{
	_copied_input_data_file_output_filename = _output_basename + "_" + _output_suffix + _file_extension; 
	_copy_input_data_file_from				= construct_valid_file_path(_input_directory, _input_filename);
	_copy_input_data_file_to				= construct_valid_file_path(_output_directory, _copied_input_data_file_output_filename);		
	if(_print_paths)
	{
		print								("-------> copy_input_data_file_from 	=\n " +  _copy_input_data_file_from);
		print								("-------> copy_input_data_file_to 		=\n " +  _copy_input_data_file_to);
	}
	File.copy								(_copy_input_data_file_from, _copy_input_data_file_to);	
	_copied_from_log						= Array.concat(_copied_from_log, _copy_input_data_file_from);				
	_copied_to_log							= Array.concat(_copied_to_log, _copy_input_data_file_to);				
	return 									_copy_input_data_file_to;
}
function copy_parameter_value_test_input_data_files(input_directory, input_filename, output_directory, output_basename, multiplot_parameter_prefix, multiplot_parameter_value_string, target_test_parameter_range_filenaming, multiplot_parameter_range_filenaming)
{
	copied_input_data_file_output_filename = output_basename + "_" + target_test_parameter_range_filenaming + "_" + multiplot_parameter_range_filenaming + "_" + multiplot_parameter_prefix + "=" + multiplot_parameter_value_string + ".txt"; 
	copy_input_data_file_from			= construct_valid_file_path(input_directory, input_filename);
	copy_input_data_file_to				= construct_valid_file_path(output_directory, copied_input_data_file_output_filename);		
	//print("copy_input_data_file_from				= " +  copy_input_data_file_from);
	//print("copy_input_data_file_to 			= " +  copy_input_data_file_to);
	File.copy								(copy_input_data_file_from, copy_input_data_file_to);					
	return copy_input_data_file_to;
}
function count_duplicate_parameter_value_reconstructions(reconstruction_directory, start_duplicate_check)
{
	duplicates = start_duplicate_check;
	if(endsWith(reconstruction_directory, File.separator))
		reconstruction_directory = substring(reconstruction_directory, 0, lengthOf(reconstruction_directory) - 1);
	duplicate_reconstruction_directory_check = reconstruction_directory + "_" + d2s(start_duplicate_check + 1, 0);
	if( File.exists(duplicate_reconstruction_directory_check) )
		duplicates = count_duplicate_parameter_value_reconstructions(reconstruction_directory, start_duplicate_check + 1);
	return duplicates;
}
function csv_2_array(path, filename, column_headings, rows, print_path)
{
	file 				= construct_valid_file_path(path, filename);		
	if(print_path)
		print			("------->Reading CSV at:\n" + file);
	data_array			= newArray(rows * column_headings.length);
	open				(file);
	for(column = 0; column < column_headings.length; column++)
		for(row = 0; row < rows; row++)
			data_array[column * rows + row] = getResult(column_headings[column], row);	
	INPUT_FILE_LIST		= Array.concat(INPUT_FILE_LIST, file);			
	return data_array;
}
function d2s_minimized_precision(value)
{
	i = 0;
	while(true)
	{
		value_string 	= d2s(value, i);
		new_value 		= parseFloat(value_string);
		if(value - new_value == 0)
			return value_string;	
		i++;
	}
} 
function elementwise_add(array, value)
{
	for(i = 0; i < array.length; i++)
		array[i] += value;
	return array;
}
function elementwise_multiply(array, value)
{
	for(i = 0; i < array.length; i++)
		array[i] *= value;
	return array;
}
function elementwise_operation(array, value_or_array, operation)
{
	num_values = value_or_array.length;
	if(operation == "add")	
	{
		for(i = 0; i < array.length; i++)
		{
			if(num_values > 1)
				array[i] += value_or_array;
			else
				array[i] += value_or_array[i];
		}
	}
	if(operation == "multiply")
	{
		for(i = 0; i < array.length; i++)
		{
			if(num_values > 1)
				array[i] *= value_or_array;
			else
				array[i] *= value_or_array[i];
		}
	}
	return array;
}
function extract_slice(slice, magnification, flip_horizontally, flip_vertically, grayscale_min, grayscale_max, overwrite_existing, window_title, close_image)
{
	print				("Extracting and saving PNG image of the slice of x being analysed...");			
	dimensions 			= query_open_image_dimensions();
	image_columns 		= dimensions[0];
	image_rows  		= dimensions[1];					
	image_slices 		= dimensions[2]; 					
	image_size_x 		= magnification * image_columns;
	image_size_y 		= magnification * image_rows;
	y_upper_LHS 		= (slice - 1) * image_rows;			// y-coordinate of upper left corner of slice to be analyzed
	makeRectangle		( 0, y_upper_LHS, image_columns, image_rows);
	//imageCalculator	("Copy create", filename ,filename );
	//run				("Enhance Contrast", "saturated=0.35");
	run					("Duplicate...", "title=" + window_title );
	setMinAndMax		(grayscale_min, grayscale_max);
	if(flip_horizontally)
		run				("Flip Horizontally");
	if(flip_vertically)
		run				("Flip Vertically");
	run					("Size...", "width=image_size_x height=image_size_y constrain average interpolation=Bilinear");	
}
function extract_subsequence_group(sequence, group_2_extract, num_groups, group_size)
{
	subsequence_group = newArray(group_size);
	for(group_element = 0; group_element < group_size; group_element++)
	{
		index = group_2_extract + group_element * num_groups;
		subsequence_group[group_element] = sequence[index];
	}
	return subsequence_group;
}
function extract_parameter_values(_target_parameter_index, _parameter_values, _num_parameter_values, _parameter_value_offsets)
{
	_target_parameter_start_index 			= _parameter_value_offsets[_target_parameter_index];
	_target_parameter_end_index 			= _parameter_value_offsets[_target_parameter_index] + _num_parameter_values[_target_parameter_index] ;
	_target_parameter_values 				= Array.slice(_parameter_values, _target_parameter_start_index, _target_parameter_end_index); 	
	return _target_parameter_values;
}
function extract_subsequence_group(sequence, group_2_extract, num_groups, group_size)
{
	subsequence_group = newArray(group_size);
	for(group_element = 0; group_element < group_size; group_element++)
	{
		index = group_2_extract + group_element * num_groups;
		subsequence_group[group_element] = sequence[index];
	}
	return subsequence_group;
}
function file_2_array(path, filename, print_path)
{
	file_path 		= construct_valid_file_path(path, filename);		
	if(print_path)
		print		("------->Importing array from:\n" + file_path);
	str 			= File.openAsString(file_path); 
	lines 			= split(str,"\n");
	array 			= newArray(lines.length);
	for(i = 0; i < array.length; i++)
		array[i] 	= lines[i];
	INPUT_FILE_LIST	= Array.concat(INPUT_FILE_LIST, file_path);			
	return array;
}
function file_2_float_array(path, filename, print_path)
{
	file_path 		= construct_valid_file_path(path, filename);		
	if(print_path)
		print		("------->Importing array from:\n" + file_path);
	str 			= File.openAsString(file_path); 
	lines 			= split(str,"\n");
	array 			= newArray(lines.length);
	for(i = 0; i < array.length; i++)
		array[i] 	= parseFloat(lines[i]);
	INPUT_FILE_LIST	= Array.concat(INPUT_FILE_LIST, file_path);			
	return array;
}
function file_2_key_value_pairs(_ROI_definitions_directory, _ROI_definitions_filename, _phantom_basename, _ROI_definitions_parameter_list, _ROI_parameter_decodings, _ROI_parameter_value_parseFloat, _print_ROI_definitions_path)
{
	_ROI_definition_file_key_value_pairs		= file_2_array(_ROI_definitions_directory, _ROI_definitions_filename, _print_ROI_definitions_path);
	_num_ROI_key_value_pairs					= _ROI_definition_file_key_value_pairs.length;
	_ordered_ROI_parameter_strings				= newArray(_num_ROI_key_value_pairs);
	for(i = 0; i < _num_ROI_key_value_pairs; i++)
	{
		separated_key_value_pair 				= split(_ROI_definition_file_key_value_pairs[i], "=");
		key_string 								= separated_key_value_pair[0];
		_spaceless_key_string 					= strip_surrounding_spaces(key_string);
		_ROI_parameter_values_string			= strip_surrounding_spaces(separated_key_value_pair[1]);
		for(j = 0; j < _ROI_definitions_parameter_list.length; j++)
			if(_spaceless_key_string == _ROI_definitions_parameter_list[j] )
				_ordered_ROI_parameter_strings[j] = _ROI_parameter_values_string;
	}	
	return _ordered_ROI_parameter_strings;	
}
function find_array_extrema(data_array, tolerance, notify_equality)
{
	extrema 			= newArray(2);
	min_val_indices 	= Array.findMinima(data_array, tolerance);
	max_val_indices 	= Array.findMaxima(data_array, tolerance);
	if(min_val_indices.length == 0)
	{
		if(notify_equality)
			print		("------->All array data values are equal");
		constant_val 	= data_array[0];									
		extrema[0] 		= constant_val;
		extrema[1] 		= constant_val;	
	}
	else
	{							
		extrema[0] 		= data_array[min_val_indices[0]];
		extrema[1] 		= data_array[max_val_indices[0]];	
	}
	return extrema;
}
function find_min_d2s_precision(value_string, max_precision)
{
	min_precision 			= 0;
	value 					= parseFloat(value_string);
	for(i=0; i< max_precision;i++)
	{
		new_value_string 	= d2s(value, i);
		new_value 			= parseFloat(new_value_string);
		if(value - new_value != 0)
			min_precision++;
	}
	return min_precision;
}
function folder_2_multiplot_string(folder, string_prefixes, multiplot_parameter, parameter_values, num_parameter_values, parameter_value_offsets, parameter_string_precisions)
{
	multiplot_string_ID 		= generate_multiplot_string(parameter_values, parameter_value_offsets, parameter_string_precisions, multiplot_parameter);
	current_parameter_values 	= folder_name_2_parameter_values (folder, string_prefixes);
	parameter_string 			= "";
	for(parameter = 0; parameter < num_parameter_values.length - 1; parameter++)
	{	
		if( parameter == multiplot_parameter - 1)
			parameter_string 	= parameter_string  + multiplot_string_ID + ", ";						
		else
			parameter_string 	= parameter_string  + string_prefixes[parameter] + "=" + current_parameter_values[parameter] + ", ";
	}	
	parameter_string 			= substring(parameter_string, 0, lengthOf(parameter_string)-2);
	return parameter_string;
}
function folder_2_parameter_string(folder, string_prefixes, string_precisions, last_parameter_2_include)
{
	current_parameter_values 	= folder_name_2_parameter_values(folder, string_prefixes);
	parameter_string 			= "";
	for(parameter = 0; parameter <= last_parameter_2_include; parameter++)
	{	
			parameter_string 	= parameter_string  + string_prefixes[parameter] + " = " + current_parameter_values[parameter] + ", ";
	}	
	parameter_string 			= substring(parameter_string, 0, lengthOf(parameter_string)-2);
	return parameter_string;
}
function folder_name_2_parameter_equals_strings(_folder, _parameter_prefixes, _parameter_string_precisions, _remove_trailing_zeros)
{
	_num_parameters = _parameter_prefixes.length;
	_parameter_equals_strings = newArray(_num_parameters);
	for(j = 0; j < _num_parameters; j++)
	{
		_prefix_start_index = indexOf(_folder, _parameter_prefixes[j]);
		_prefix_end_index = _prefix_start_index + lengthOf(_parameter_prefixes[j]) + 1;
		_parameter_value_end_index = indexOf(_folder, "_", _prefix_end_index);
		if(_parameter_value_end_index == -1) 
			_parameter_value_end_index = lengthOf(_folder);
		_parameter_value_string = substring(_folder, _prefix_end_index, _parameter_value_end_index);
		if(_remove_trailing_zeros)
			_parameter_value_string = remove_trailing_zeros(_parameter_value_string, _parameter_string_precisions[j]);
		_parameter_value_equal_string = _parameter_prefixes[j] + "=" + _parameter_value_string;		
		_parameter_equals_strings[j] =  _parameter_value_equal_string;
	}
	return _parameter_equals_strings;
}
function folder_name_2_parameter_values(_folder_name, _parameter_string_prefixes)
{
	_num_parameters 		= _parameter_string_prefixes.length;
	_prefix_lengths 		= newArray		(_parameter_string_prefixes.length);
	_value_strings 			= newArray		(_num_parameters);
	for( i = 0; i < _num_parameters - 1; i++)
	{
		_prefix 			= _parameter_string_prefixes[i];
		_prefix_length 		= lengthOf		(_prefix);
		_start_index 		= indexOf		(_folder_name, _prefix );
		_end_index 			= indexOf		(_folder_name, "_", _start_index + _prefix_length + 1);
		_value_strings[i] 	= substring		(_folder_name, _start_index + _prefix_length + 1, _end_index);
	}
	_next_prefix 			= indexOf		(_folder_name, _parameter_string_prefixes[_num_parameters - 1] );
	_start_index 			= indexOf		(_folder_name, "_", _next_prefix + 1);
	_end_index 				= indexOf		(_folder_name, "_", _start_index + 1);
	if(_end_index == -1)
		_end_index = lengthOf(_folder_name);
	_value_strings[i] 		= substring		(_folder_name, _start_index + 1, _end_index );
	return _value_strings;
}
function generate_all_parameter_value_strings(_parameter_values, _num_parameter_values, _parameter_value_offsets, _parameter_string_precisions)
{
	_num_parameters 			= _parameter_string_precisions.length;
	_parameter_value_strings	= newArray(_parameter_values.length);
	index 						= 0;
	for(parameter = 0; parameter < _num_parameters; parameter++)
		for(i = 0; i < _num_parameter_values[parameter]; i++)
			_parameter_value_strings[index++]	= d2s(_parameter_values[i + _parameter_value_offsets[parameter]], _parameter_string_precisions[parameter]);
	return _parameter_value_strings;
}
function generate_averaged_data_CSV(reconstruction_data_directory, base_reconstruction_folder, input_data_subdirectory, input_data_filename, input_CSV_columns, input_CSV_rows, ROIs_per_reconstruction)
{
	print_section_separator("Generating average quality measures from the ROI analysis CSV files from each reconstruction performed with the current parameters", false);
	all_RSP_data 					= newArray();
	current_test_input_directory 	= construct_valid_directory_path(reconstruction_data_directory, base_reconstruction_folder);
	while(startsWith(input_data_filename , File.separator))
		input_data_filename	 		= substring(input_data_filename, 1, lengthOf(input_data_filename));
	duplicates 						= count_duplicate_parameter_value_reconstructions(current_test_input_directory, 0);
	avg_RSP_data 					= newArray(ROIs_per_reconstruction * (duplicates + 1) );
	base_folder_data 				= csv_2_array(current_test_input_directory + input_data_subdirectory, input_data_filename, input_CSV_columns, input_CSV_rows, false);
	all_RSP_data 					= Array.concat(all_RSP_data, base_folder_data);		
	for(duplicate = 1; duplicate <= duplicates; duplicate++)
	{
		folder 						= base_reconstruction_folder + "_" + d2s(duplicate, 0);
		directory 					= construct_valid_directory_path(reconstruction_data_directory, folder);
		RSP_data 					= csv_2_array(directory  + input_data_subdirectory, input_data_filename, input_CSV_columns, input_CSV_rows, false);
		all_RSP_data 				= Array.concat(all_RSP_data, RSP_data);
	}
	avg_RSP_data 					= newArray(ROIs_per_reconstruction);
	avg_RSP_data_strings 			= newArray(ROIs_per_reconstruction);
	Array.fill(avg_RSP_data, 0);
	for(i = 0; i < ROIs_per_reconstruction; i++)
	{
		for(duplicate = 0; duplicate <= duplicates; duplicate++)
			avg_RSP_data[i] 		+= all_RSP_data[i + duplicate * ROIs_per_reconstruction];
		avg_RSP_data[i] 			/= (duplicates + 1);
	}						
	for(i = 0; i < ROIs_per_reconstruction; i++)
		avg_RSP_data_strings[i] 	= d2s(avg_RSP_data[i], 10);
	make_top_directory 				= true;
	_print_directories_created 		= true;
	overwrite_averaged_data_CSV 	= true;
	print_averaged_data_CSV_path 	= true;
	averaged_data_output_directory 	= construct_valid_directory_path(reconstruction_data_directory, base_reconstruction_folder + averaged_data_folder_suffix + input_data_subdirectory);
	parents 						= make_directory_parents(reconstruction_data_directory, averaged_data_output_directory, folder_separator, make_top_directory, _print_directories_created);
	array_2_CSV						(num_ROIs_2_analyze, images_per_reconstruction, avg_RSP_data_strings, averaged_data_output_directory, input_data_filename, print_averaged_data_CSV_path, overwrite_averaged_data_CSV);	
	return 							avg_RSP_data_strings;			
}
function generate_averaged_data_TXT(reconstruction_data_directory, base_reconstruction_folder, input_data_filename, measurements_per_reconstruction)
{
	print_section_separator("Generating average quality measures from the ROI analysis TXT files from each reconstruction performed with the current parameters", false);
	print_input_data_path 			= true;
	make_top_directory 				= true;
	_print_directories_created 		= true;
	overwrite_averaged_data_TXT		= true;
	print_averaged_data_TXT_path 	= true;
	all_input_data 					= newArray();
	current_test_input_directory 	= construct_valid_directory_path(reconstruction_data_directory, base_reconstruction_folder);
	while(startsWith(input_data_filename , File.separator))
		input_data_filename	 		= substring(input_data_filename, 1, lengthOf(input_data_filename));
	duplicates 						= count_duplicate_parameter_value_reconstructions(current_test_input_directory, 0);
	print("\nDuplicate reconstructions found = " + duplicates);
	avg_input_data 					= newArray(measurements_per_reconstruction * (duplicates + 1) );
	base_folder_data 				= file_2_array(current_test_input_directory, input_data_filename, print_input_data_path);
	all_input_data 					= Array.concat(all_input_data, base_folder_data);		
	for(duplicate = 1; duplicate <= duplicates; duplicate++)
	{
		print("Adding duplicate # " + duplicate + " data to averaged data file ");
		folder 						= base_reconstruction_folder + "_" + d2s(duplicate, 0);
		directory 					= construct_valid_directory_path(reconstruction_data_directory, folder);
		input_data 					= file_2_array(directory, input_data_filename, print_input_data_path);
		all_input_data 				= Array.concat(all_input_data, input_data);
	}
	Array.print(all_input_data);
	avg_input_data					= newArray(measurements_per_reconstruction);
	//avg_input_data_strings 			= newArray(measurements_per_reconstruction);
	Array.fill(avg_input_data, 0);
	for(i = 0; i < measurements_per_reconstruction; i++)
	{
		//all_input_data[i]				= parseFloat(all_input_data[i]);
		for(duplicate = 0; duplicate <= duplicates; duplicate++)
			avg_input_data[i] 		= avg_input_data[i] + all_input_data[i + duplicate * measurements_per_reconstruction];
		avg_input_data[i] 			/= (duplicates + 1);
		avg_input_data_strings 		= d2s(avg_input_data[i], 10);
	}
	averaged_data_output_directory 	= construct_valid_directory_path(reconstruction_data_directory, base_reconstruction_folder + averaged_data_folder_suffix);
	parents 						= make_directory_parents(reconstruction_data_directory, averaged_data_output_directory, folder_separator, make_top_directory, _print_directories_created);
	array_2_TXT						(averaged_data_output_directory, input_data_filename, avg_input_data, print_averaged_data_TXT_path, overwrite_averaged_data_TXT);	
	print("========>Finished calculating and writing averaged reconstruction data to disk at:\n" + averaged_data_output_directory);
	return 							avg_input_data;			
}
function generate_combined_filename_suffixes(_leading_filename_suffixes, _trailing_filename_suffixes)
{
	_num_leading_filename_suffixes 	= _leading_filename_suffixes.length;
	_num_trailing_filename_suffixes	= _trailing_filename_suffixes.length;
	_combined_filename_suffixes 	= newArray(_num_leading_filename_suffixes * _num_trailing_filename_suffixes);
	for(i = 0; i < _num_leading_filename_suffixes; i++)
		for(j = 0; j < _num_trailing_filename_suffixes; j++)
			_combined_filename_suffixes[i * _num_trailing_filename_suffixes + j] = _leading_filename_suffixes[i] + "_" + _trailing_filename_suffixes[j];
	return _combined_filename_suffixes;
}
function generate_combined_parameter_value_range_filenames(_parameter_values, _num_parameter_values, _parameter_value_offsets, _parameter_string_prefixes, _parameter_string_precisions)
{
	_parameter_value_test_identifier 			= "";
	_num_parameters 							= _num_parameter_values.length;
	for(parameter = 0; parameter < _num_parameters; parameter++)
	{
		_range_min_string 						= remove_trailing_zeros(_parameter_values[_parameter_value_offsets[parameter]], _parameter_string_precisions[parameter]);
		_range_max_string 						= remove_trailing_zeros(_parameter_values[_parameter_value_offsets[parameter] + _num_parameter_values[parameter] - 1], _parameter_string_precisions[parameter]);	
		_prefix 								= _parameter_string_prefixes[parameter];
		if(_num_parameter_values[parameter] == 1 )
			_parameter_value_test_identifier 	+= _prefix + "[" + _range_min_string + "]_";
		else
			_parameter_value_test_identifier 	+= _prefix + "[" + _range_min_string + "-" + _range_max_string + "]_";			
	}
	//_parameter_value_test_identifier 			+= _parameter_string_prefixes[_num_parameters - 1] + _target_test_parameter_range_suffix;	
	_parameter_value_test_identifier			= substring(_parameter_value_test_identifier, 0, lengthOf(_parameter_value_test_identifier) - 1);
	return _parameter_value_test_identifier;						
}
function generate_combined_parameter_value_range_string(_parameter_values, _num_parameter_values, _parameter_value_offsets, _parameter_string_prefixes, _parameter_string_precisions)
{
	_parameter_value_test_identifier 			= "";
	_num_parameters 							= _num_parameter_values.length;
	for(parameter = 0; parameter < _num_parameters; parameter++)
	{
		_range_min_string 						= remove_trailing_zeros(_parameter_values[_parameter_value_offsets[parameter]], _parameter_string_precisions[parameter]);
		_range_max_string 						= remove_trailing_zeros(_parameter_values[_parameter_value_offsets[parameter] + _num_parameter_values[parameter] - 1], _parameter_string_precisions[parameter]);	
		_prefix 								= _parameter_string_prefixes[parameter];
		if(_num_parameter_values[parameter] == 1 )
			_parameter_value_test_identifier 	+= _prefix + "[" + _range_min_string + "]_";
		else
			_parameter_value_test_identifier 	+= _prefix + "[" + _range_min_string + "-" + _range_max_string + "]_";			
	}
	//_parameter_value_test_identifier 			+= _parameter_string_prefixes[_num_parameters - 1] + _target_test_parameter_range_suffix;	
	_parameter_value_test_identifier			= substring(_parameter_value_test_identifier, 0, lengthOf(_parameter_value_test_identifier) - 1);
	return _parameter_value_test_identifier;						
}
function generate_current_parameter_value_test_multiplot_output_directory(_test_batch_directory, _multiplot_parameter_index, _current_parameter_value_test_value_strings, _multiplot_suffix_string, _target_test_parameter_range_suffix, _parameter_string_prefixes, _parameter_string_precisions)
{
	_current_parameter_value_test_multiplot_output_directory 		= _test_batch_directory;
	_num_parameters					= _parameter_string_prefixes.length;		
	for(parameter = 0; parameter < _multiplot_parameter_index; parameter++)	
		_current_parameter_value_test_multiplot_output_directory	+= parameter_string_prefixes[parameter] + "_" + _current_parameter_value_test_value_strings[parameter] + "_";	
	_current_parameter_value_test_multiplot_output_directory		+= _multiplot_suffix_string + "_";						
	for(parameter = _multiplot_parameter_index + 1; parameter < num_parameters - 1; parameter++)	
		_current_parameter_value_test_multiplot_output_directory 	+= parameter_string_prefixes[parameter] + "_" + _current_parameter_value_test_value_strings[parameter] + "_";						
	_current_parameter_value_test_multiplot_output_directory	+=  parameter_string_prefixes[num_parameters - 1] + _target_test_parameter_range_suffix;			
	return _current_parameter_value_test_multiplot_output_directory;	
}
function generate_current_parameter_value_test_parameter_values_string(_current_parameter_value_test_value_strings, _target_test_parameter_index, _parameter_string_prefixes)
{
	_num_parameters					= _parameter_string_prefixes.length;		
	_parameter_values_string 		= "";
	for(parameter = 0; parameter < _num_parameters; parameter++)
		_parameter_values_string 	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";
	_parameter_values_string 		= substring(_parameter_values_string, 0, lengthOf(_plot_title_parameter_string) - 2);										
	return _parameter_values_string;
}
function generate_legend_entries(common_entry_text, entry_specific_text)
{
	legend_entries			= newArray(lengthOf(entry_specific_text));
	for(i = 0; i < lengthOf(entry_specific_text); i++)
		legend_entries[i]	= common_entry_text + entry_specific_text[i];
	return legend_entries;
}
function generate_missing_input_data(common_directory, current_parameter_value_test_folder, RSP_path, RSP_Error_path, Std_Dev_path)
{
	directory 		= construct_valid_directory_path(common_directory, current_parameter_value_test_folder);		
	data_verified 		= true;
	if(!File.exists(directory + RSP_path))
	{	
		print			("------->Missing RSP input file:\n" + directory + RSP_path);
		data_verified 	= false;
	}
	if(!File.exists(directory + RSP_Error_path))
	{	
		print			("------->Missing RSP error input file:\n" + directory + RSP_Error_path);
		data_verified 	= false;
	}
	if(!File.exists(directory + Std_Dev_path))
	{	
		print			("------->Missing std dev input file:\n" + directory + Std_Dev_path);
		data_verified 	= false;
	}
	if(!data_verified)
	{
		if(endsWith(directory, File.separator))
			runMacro	(ROI_analysis_macro_path, directory);
		else
			runMacro	(ROI_analysis_macro_path, directory + folder_separator);	
	}			
	return true;					
}		
function generate_missing_test_input_data(common_directory, current_parameter_value_test_folders, RSP_path, RSP_Error_path, Std_Dev_path)
{
	for( i = 0; i < current_parameter_value_test_folders.length; i++)
	{
		directory 			= construct_valid_directory_path(common_directory, current_parameter_value_test_folders[i]);		
		data_verified 		= true;
		print				("------->Path to RSP input file:\n" + directory + RSP_path);
		print				("------->Path to RSP error input file:\n" + directory + RSP_Error_path);
		print				("------->Path to standard deviation input file:\n" + directory + Std_Dev_path);
		if(!File.exists(directory + RSP_path))
		{	
			print			("------->Missing RSP input file:\n" + directory + RSP_path);
			data_verified 	= false;
		}
		if(!File.exists(directory + RSP_Error_path))
		{	
			print			("------->Missing RSP error input file:\n" + directory + RSP_Error_path);
			data_verified 	= false;
		}
		if(!File.exists(directory + Std_Dev_path))
		{	
			print			("------->Missing standard deviation input file:\n" + directory + Std_Dev_path);
			data_verified 	= false;
		}
		if(!data_verified)
		{
			if(endsWith(directory, File.separator))
				runMacro	(ROI_analysis_macro_path, directory);
			else
				runMacro	(ROI_analysis_macro_path, directory + folder_separator);	
		}
	}
	return true;			
}		
function generate_modulo_values(num_parameter_values)
{
	num_parameters				= num_parameter_values.length;
	modulo_values 				= newArray(num_parameters - 1);	
	Array.fill					(modulo_values, 1);
	for(i = 0; i < num_parameters; i++)
		for(j = i + 1; j < num_parameters; j++)
			modulo_values[i] 	*= num_parameter_values[j];		 
	return modulo_values;
}
function generate_multiplot_parameter_set_string(_multiplot_parameter_index, _num_multiplot_parameter_values, _parameter_values, _parameter_value_offsets, _parameter_string_prefixes, _parameter_string_precisions)
{
	_multiplot_suffix_string 		= _parameter_string_prefixes[_multiplot_parameter_index] + "[";
	for(i = 0; i < _num_multiplot_parameter_values; i++)
		_multiplot_suffix_string 	= _multiplot_suffix_string  + d2s(_parameter_values[_parameter_value_offsets[_multiplot_parameter_index] + i], _parameter_string_precisions[_multiplot_parameter_index]) + ",";
	_multiplot_suffix_string 		= substring(_multiplot_suffix_string, 0, lengthOf(_multiplot_suffix_string) - 1) + "]";
	return _multiplot_suffix_string;
}
function generate_multiplot_string(parameter_values, parameter_value_offsets, parameter_string_precisions, multiplot_parameter)
{
	multiplot_string_ID 		= parameter_string_prefixes[multiplot_parameter_index] + " = [";
	for(i = 0; i < num_multiplot_parameter_values; i++)
		multiplot_string_ID 	= multiplot_string_ID  + d2s(parameter_values[parameter_value_offsets[multiplot_parameter_index] + i], parameter_string_precisions[multiplot_parameter_index]) + ",";
	multiplot_string_ID 		= substring(multiplot_string_ID, 0, lengthOf(multiplot_string_ID) - 1) + "]";
	return multiplot_string_ID;
}
function generate_multiplot_suffix_string(_multiplot_parameter_index, _num_multiplot_parameter_values, _parameter_values, _parameter_value_offsets, _parameter_string_prefixes, _parameter_string_precisions)
{
	_multiplot_suffix_string 		= _parameter_string_prefixes[_multiplot_parameter_index] + "[";
	for(i = 0; i < _num_multiplot_parameter_values; i++)
		_multiplot_suffix_string 	= _multiplot_suffix_string  + d2s(_parameter_values[_parameter_value_offsets[_multiplot_parameter_index] + i], _parameter_string_precisions[_multiplot_parameter_index]) + ",";
	_multiplot_suffix_string 		= substring(_multiplot_suffix_string, 0, lengthOf(_multiplot_suffix_string) - 1) + "]";
	return _multiplot_suffix_string;
}
function generate_multiplot_title_parameter_string(_target_test_parameter_range_suffix, _multiplot_parameter_index, _current_parameter_value_test_value_strings, _multiplot_parameter_range_plots, _parameter_string_prefixes )
{
	_multiplot_title_parameter_string 		= "";
	_num_parameters					= _parameter_string_prefixes.length;		
	for(parameter = 0; parameter < _multiplot_parameter_index; parameter++)	
		_multiplot_title_parameter_string	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";	
	_multiplot_title_parameter_string		+= _multiplot_parameter_range_plots + ", ";						
	for(parameter = _multiplot_parameter_index + 1; parameter < num_parameters - 1; parameter++)	
		_multiplot_title_parameter_string 	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";						
	_multiplot_title_parameter_string	+=  parameter_string_prefixes[num_parameters - 1] + " = " + _target_test_parameter_range_suffix;			
	return _multiplot_title_parameter_string;	
}		
function generate_multiplot_title_parameter_values_string(_target_test_parameter_range_suffix, _multiplot_parameter_index, _current_parameter_value_test_value_strings, _multiplot_parameter_range_filenaming, _parameter_string_prefixes )
{
	_multiplot_title_parameter_values_string 		= "";
	_num_parameters					= _parameter_string_prefixes.length;		
	for(parameter = 0; parameter < _multiplot_parameter_index; parameter++)	
		_multiplot_title_parameter_values_string	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";	
	_multiplot_title_parameter_values_string		+= _multiplot_parameter_range_filenaming + ", ";						
	for(parameter = _multiplot_parameter_index + 1; parameter < num_parameters - 1; parameter++)	
		_multiplot_title_parameter_values_string 	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";						
	_multiplot_title_parameter_values_string	+=  parameter_string_prefixes[num_parameters - 1] + " = " + _target_test_parameter_range_suffix;			
	return _multiplot_title_parameter_values_string;	
}		
function generate_multivalue_parameter_equals_suffix(_current_test_parameter_value_strings, _parameter_prefixes, _num_parameter_values, _parameter_precisions, _multiplot_parameter_index, _multiplot_parameter_range_filenaming)
{
	_suffix = "";
	_num_parameters = _parameter_prefixes.length;
	for(i = 0; i < _num_parameters - 1; i++)
	{
		if(_num_parameter_values[i] > 1)
		{
			if(i != _multiplot_parameter_index)
				_suffix += _parameter_prefixes[i] + "=" + remove_trailing_zeros(_current_test_parameter_value_strings[i], _parameter_precisions[i]) + "_";
			else
				_suffix += _multiplot_parameter_range_filenaming + "_";
		}
		
	}
	return substring(_suffix, 0, lengthOf(_suffix) -1);						
}
function generate_multivalue_parameter_values_short_title_string(_target_test_parameter_range_index, _multiplot_parameter_index, _current_parameter_value_test_value_strings, _parameter_string_prefixes, _num_parameter_values )
{
	_multiplot_title_parameter_values_string 		= "";
	_num_parameters					= _parameter_string_prefixes.length;		
	for(parameter = 0; parameter < _num_parameters; parameter++)	
	{
		if(_num_parameter_values[parameter] > 1 && parameter != _target_test_parameter_range_index && parameter != _multiplot_parameter_index)
			_multiplot_title_parameter_values_string	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";	
	}
	return _multiplot_title_parameter_values_string;	
}		
function generate_multivalue_parameter_values_title_string(_target_test_parameter_range_suffix, _multiplot_parameter_index, _current_parameter_value_test_value_strings, _multiplot_parameter_range_filenaming, _parameter_string_prefixes, _num_parameter_values )
{
	_multiplot_title_parameter_values_string 		= "";
	_num_parameters					= _parameter_string_prefixes.length;		
	for(parameter = 0; parameter < _multiplot_parameter_index; parameter++)	
		if(_num_parameter_values[parameter] > 1)
			_multiplot_title_parameter_values_string	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";	
	_multiplot_title_parameter_values_string		+= _multiplot_parameter_range_filenaming + ", ";						
	for(parameter = _multiplot_parameter_index + 1; parameter < num_parameters - 1; parameter++)	
		if(_num_parameter_values[parameter] > 1)
			_multiplot_title_parameter_values_string 	+= parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";						
	_multiplot_title_parameter_values_string	+=  parameter_string_prefixes[num_parameters - 1] + " = " + _target_test_parameter_range_suffix;			
	return _multiplot_title_parameter_values_string;	
}		
function generate_ordering_indices(_num_iterations, _num_ROIs_2_analyze, _images_per_reconstruction, _num_target_test_parameter_values)
{
	ordering_indices 										= newArray(_num_ROIs_2_analyze * _images_per_reconstruction * _num_target_test_parameter_values);
	ordering_indices_position 								= 0;
	for(iteration = 0; iteration <= _num_iterations; iteration++)  
	{
		for(ROI = 0; ROI < _num_ROIs_2_analyze; ROI++)
		{
			for( N = 0; N < _num_target_test_parameter_values; N++)
			{
				ordering_index 								= _num_ROIs_2_analyze * _images_per_reconstruction * N + iteration * _num_ROIs_2_analyze + ROI;
				ordering_indices[ordering_indices_position] = ordering_index;
				ordering_indices_position++;
			}
		}
	}
	return ordering_indices;										
}
function generate_parameter_test_combination_indices(_target_test_parameter_index, _combination_index, _modulo_values, _parameter_value_offsets )
{	
	j 								= _combination_index;
	_parameter_combination_indices 		= newArray(_target_test_parameter_index + 1);
	for(parameter = 0; parameter < _target_test_parameter_index; parameter++)
	{
		i = 0;
		while(j - _modulo_values[parameter] >= 0)
		{
			j = j - _modulo_values[parameter];
			i++;
		}
		_parameter_combination_indices[parameter]					= i + _parameter_value_offsets[parameter];
	}		
	_parameter_combination_indices[_target_test_parameter_index]	= j + _parameter_value_offsets[_target_test_parameter_index];		
	return _parameter_combination_indices;       	
}
function generate_parameter_test_combination(_parameter_combination_indices, _parameter_values)
{
	num_parameter_combination_indices 	= _parameter_combination_indices.length;
	_parameter_combination 				= newArray(num_parameter_combination_indices);
	for(parameter = 0; parameter < num_parameter_combination_indices; parameter++)
		_parameter_combination[parameter]	= _parameter_values[_parameter_combination_indices[parameter]];					
	return _parameter_combination;       	
}
function generate_parameter_test_folder(_parameter_string_prefixes, _parameter_string_precisions, _parameter_combination )
{
	_num_parameters					= _parameter_string_prefixes.length;
	_folder_string 					= "";
	for(parameter = 0; parameter < _num_parameters; parameter++)	
		_folder_string 				= _folder_string + "_" + _parameter_string_prefixes[parameter] + "_" + d2s( _parameter_combination[parameter], _parameter_string_precisions[parameter]);	
	_folder_string					= substring(_folder_string, 1);
	return _folder_string;	
}
function generate_parameter_test_folder_compared(_parameter_string_prefixes, _parameter_string_precisions, _parameter_combination )
{
	_num_parameters					= _parameter_string_prefixes.length;
	_folder_string 					= "";
	for(parameter = 0; parameter < _num_parameters; parameter++)	
		_folder_string 				= _folder_string + "_" + _parameter_string_prefixes[parameter] + "_" + d2s( _parameter_combination[parameter], _parameter_string_precisions[parameter]);	
	_folder_string					= substring(_folder_string, 1);
	return _folder_string;	
}
function generate_parameter_test_multiplot_folder(_multiplot_parameter_index, _multiplot_parameter_combination, _multiplot_combination_offset, _parameter_values, _parameter_value_offsets, _parameter_string_prefixes, _parameter_string_precisions )
{
	_num_parameters					= _parameter_string_prefixes.length;		
	_folder_string 					= "";
	for(parameter = 0; parameter < _multiplot_parameter_index; parameter++)	
		_folder_string	= _folder_string + "_" + _parameter_string_prefixes[parameter] + "_" + d2s( _multiplot_parameter_combination[parameter], _parameter_string_precisions[parameter]);
	_folder_string		= _folder_string + "_" + _parameter_string_prefixes[parameter] + "_" + d2s( _parameter_values[_parameter_value_offsets[parameter] + _multiplot_combination_offset], _parameter_string_precisions[parameter]);	
	for(parameter = _multiplot_parameter_index + 1; parameter < num_parameters; parameter++)	
		_folder_string 	= _folder_string + "_" + _parameter_string_prefixes[parameter] + "_" + d2s( _multiplot_parameter_combination[parameter - 1], _parameter_string_precisions[parameter]);
	_folder_string	= substring(_folder_string, 1);				
	return _folder_string;	
}		
function generate_parameter_value_offsets(num_parameter_values)
{
	num_parameters				= num_parameter_values.length;
	parameter_value_offsets 	= newArray(num_parameters);	
	for(i = 0; i < num_parameters; i++)
		for(j = 0; j < i; j++)
			parameter_value_offsets[i] 	+= num_parameter_values[j];		 
	return parameter_value_offsets;
}
function generate_parameter_value_range_suffix_string(_parameter_values, _parameter_string_precision)
{	
	_target_test_parameter_extrema = find_array_extrema(_parameter_values, tolerance, false);
	_target_test_parameter_range_min_string		= d2s( _target_test_parameter_extrema[0], _parameter_string_precision);
	_target_test_parameter_range_max_string		= d2s( _target_test_parameter_extrema[1], _parameter_string_precision);
	_target_test_parameter_range_suffix			= "[" + _target_test_parameter_range_min_string + "-" + _target_test_parameter_range_max_string + "]";		
	return _target_test_parameter_range_suffix;
}							
function generate_parameter_value_strings(_parameter_values, _parameter_string_precision, minimize_string_length)
{
	_num_parameters 			= _parameter_values.length;
	_parameter_value_strings	= newArray(_num_parameters);
	for(i = 0; i < _num_parameters; i++)
	{
		parameter_value_string = d2s(_parameter_values[i], _parameter_string_precision);
		if(minimize_string_length)
			parameter_value_string = remove_trailing_zeros(parameter_value_string, _parameter_string_precision);
		_parameter_value_strings[i]	= parameter_value_string;
	}
	return _parameter_value_strings;
}
function generate_parameter_value_test_multiplot_copy_suffixes(_current_test_parameter_value_strings, _parameter_prefixes, _num_parameter_values, _parameter_precisions, _multiplot_parameter_index, _multiplot_parameter_range_filenaming)
{
	_suffix = "";
	_num_parameters = _parameter_prefixes.length;
	for(i = 0; i < _num_parameters - 1; i++)
	{
		if(_num_parameter_values[i] > 1)
		{
			if(i != _multiplot_parameter_index)
				_suffix += _parameter_prefixes[i] + "=" + remove_trailing_zeros(_current_test_parameter_value_strings[i], _parameter_precisions[i]) + "_";
			else
				_suffix += _multiplot_parameter_range_filenaming + "_";
		}
		
	}
	return substring(_suffix, 0, lengthOf(_suffix) -1);						
}
function generate_plot_title_parameter_string(_current_parameter_value_test_value_strings, _target_test_parameter_index, _target_test_parameter_range_suffix, _parameter_string_prefixes)
{
	_plot_title_parameter_string 		= "";
	for(parameter = 0; parameter < _target_test_parameter_index; parameter++)
		_plot_title_parameter_string 	+= _parameter_string_prefixes[parameter] + " = " + _current_parameter_value_test_value_strings[parameter] + ", ";
	_plot_title_parameter_string 		+= _parameter_string_prefixes[parameter] + " = " + _target_test_parameter_range_suffix;										
	return _plot_title_parameter_string;
}					
function generate_reduced_modulo_values(num_parameter_values, modulo_values, skip_index)
{
	num_modulo_values 							= modulo_values.length;
	//print(num_modulo_values);
	modulo_value_divisor						= num_parameter_values[skip_index];
	reduced_modulo_values 						= newArray(num_modulo_values - 1);	
	Array.fill									(reduced_modulo_values, 1);
	for(i = 0; i < num_modulo_values; i++)
	{
		if(i < skip_index)
			reduced_modulo_values[i] 			= modulo_values[i] / modulo_value_divisor;
		if(i >= skip_index)
			if(i < num_modulo_values)
				reduced_modulo_values[i - 1] 	= modulo_values[i];			
	}
	return reduced_modulo_values;
}
function generate_reduced_parameter_value_offsets(num_parameter_values, parameter_value_offsets, skip_index)
{
	num_parameters					= num_parameter_values.length;
	reduced_parameter_value_offsets = newArray(num_parameters - 1);	
	Array.fill						(reduced_parameter_value_offsets, 1);
	for(i = 0; i < num_parameters; i++)
	{
		if(i < multiplot_parameter_index)	
			reduced_parameter_value_offsets[i] 		= parameter_value_offsets[i];
		if(i >= multiplot_parameter)
			reduced_parameter_value_offsets[i - 1]	= parameter_value_offsets[i] - num_multiplot_parameter_values;
	}
	return reduced_parameter_value_offsets;
}
function identify_missing_input_data(common_directory, current_parameter_value_test_folder, RSP_path, RSP_Error_path, Std_Dev_path)
{
	directory 			= construct_valid_directory_path(common_directory, current_parameter_value_test_folder);		
	data_verified 		= true;
	if(!File.exists(directory + RSP_path))
	{	
		print			("------->Missing RSP input file:\n" + directory + RSP_path);
		data_verified 	= false;
	}
	if(!File.exists(directory + RSP_Error_path))
	{	
		print			("------->Missing RSP error input file:\n" + directory + RSP_Error_path);
		data_verified 	= false;
	}
	if(!File.exists(directory + Std_Dev_path))
	{	
		print			("------->Missing standard deviation input file:\n" + directory + Std_Dev_path);
		data_verified 	= false;
	}
	return data_verified;			
}	
function identify_missing_test_input_data(common_directory, current_parameter_value_test_folders, RSP_path, RSP_Error_path, Std_Dev_path)
{
	for( i = 0; i < current_parameter_value_test_folders.length; i++)
	{
		directory 			= construct_valid_directory_path(common_directory, current_parameter_value_test_folders[i]);		
		data_verified 		= true;
		if(!File.exists(directory + RSP_path))
		{	
			print			("------->Missing RSP input file:\n" + directory + RSP_path);
			data_verified 	= false;
		}
		if(!File.exists(directory + RSP_Error_path))
		{	
			print			("------->Missing RSP error input file:\n" + directory + RSP_Error_path);
			data_verified 	= false;
		}
		if(!File.exists(directory + Std_Dev_path))
		{	
			print			("------->Missing standard deviation input file:\n" + directory + Std_Dev_path);
			data_verified 	= false;
		}
	}
	return true;			
}																																																																				
function image_set_2_animated_GIF(image_type, directory, filenames, image_names_matching, GIF_filename, GIF_speed, print_image_paths, close_image_set, GIF_2_disk, overwrite_GIF, print_GIF_path, close_stack_after_saving)
{
	stack_dimensions 		= image_set_2_stack(image_type, directory, filenames, image_names_matching, GIF_filename, print_image_paths, close_image_set);
	stack_2_animated_GIF	(GIF_filename, directory, GIF_filename, GIF_speed, GIF_2_disk, overwrite_GIF, print_GIF_path, close_stack_after_saving);
	return stack_dimensions;
}			
function image_set_2_AVI(image_type, directory, filenames, image_names_matching, filename, compression, frame_rate, print_image_paths, close_image_set, AVI_2_disk, overwrite_AVI, print_path, close_stack_after_saving)
{
	stack_dimensions 	= image_set_2_stack(image_type, directory, filenames, image_names_matching, filename, print_image_paths, close_image_set);
	stack_2_AVI			(filename, directory, filename, compression, frame_rate, AVI_2_disk, overwrite_AVI, print_path, close_stack_after_saving);
	return stack_dimensions;
}
function image_set_2_stack(image_type, directory, filenames, image_names_matching, stack_title, print_image_paths, close_image_set)
{
	image_set			= open_image_set(image_type, directory, filenames, print_image_paths);
	stack_dimensions 	= opened_image_set_2_stack(image_names_matching, stack_title, close_image_set);
	if(close_image_set)
		for(i = 0; i < image_set.length; i++)
			close		(image_set[i]);
	return stack_dimensions;
}	
function import_multiplot_comparison_data(_test_batch_directory, _source_subdirectory, _source_filename, _file_extension, _CSV_column_headings, _CSV_num_rows)
{
	_path 						= construct_valid_directory_path(_test_batch_directory, _source_subdirectory);
	_file_path 					= construct_valid_file_path(_path, _source_filename);
	if(File.exists(_file_path))
	{	
		print					("------->Reading compared data file = " + _source_filename);
		print					("           from:\n" + _path);
		if(_file_extension == ".txt")
			_compared_data		=  file_2_array(_path, _source_filename, false);
		else if(_file_extension == ".csv")
			_compared_data		=  csv_2_array(_path, _source_filename, _CSV_column_headings, _CSV_num_rows, false);		
		else
		{
			print				("****ERROR****No procedure defined for specified file type."); 
			_compared_data 		= newArray();
		}
		return _compared_data;
	}
	else
	{
		print					("****ERROR****: Compared data not found at:\n" + _file_path);				
		return newArray();		
	}
}
function import_multiplot_comparison_data_CSV(_test_batch_directory, _CSV_source_subdirectory, _CSV_source_filename, _CSV_column_headings, _CSV_num_rows)
{
	_path 								= construct_valid_directory_path(_test_batch_directory, _CSV_source_subdirectory);
	_file_path 							= construct_valid_file_path(_path, _CSV_source_filename);
	if(File.exists(_file_path))
	{	
		print("------->Reading compared data file = " + _CSV_source_filename);
		print("           from:\n" + _path);
		_compared_data					=  csv_2_array(_path, _CSV_source_filename, _CSV_column_headings, _CSV_num_rows, false);
		return _compared_data;
	}
	else
	{
		print("****ERROR****: Compared data not found at:\n" + _file_path);				
		return newArray();		
	}
}
function import_multiplot_comparison_data_TXT(_test_batch_directory, _TXT_source_subdirectory, _TXT_source_filename)
{
	_path 								= construct_valid_directory_path(_test_batch_directory, _TXT_source_subdirectory);
	_file_path 							= construct_valid_file_path(_path, _TXT_source_filename);
	if(File.exists(_file_path))
	{	
		print("------->Reading compared data file = " + _TXT_source_filename);
		print("           from:\n" + _path);		
		_compared_data					=  file_2_array(_path, _TXT_source_filename, false);
		return _compared_data;
	}
	else
	{
		print("****ERROR****: Compared data not found at:\n" + _file_path);				
		return newArray();		
	}
}					
function line_profile(_ROI_number, _ROI_x, _ROI_y, _ROI_radius, _profile_radius, _profile_direction, _base_value, _ref_value, _voxel_dimensions, _title, _text, _text_size, _xlabel, _ylabel, _legend_entries, _legend_text_size, _plot_parameters, _is_gradient_analysis)
{
							
	_voxel_width 				= _plot_parameters[0]; 	_tolerance		= _plot_parameters[1]; 	_lower_limits_scale	= _plot_parameters[2]; _upper_limits_scale 	= _plot_parameters[3]; 
	_difference_scale 			= _plot_parameters[4]; 	_x_frame_size  	= _plot_parameters[5]; 	_y_frame_size 		= _plot_parameters[6]; _text_xpos_ratio 		= _plot_parameters[7]; 
	_text_ypos_ratio 			= _plot_parameters[8]; 	_add_line_color 	= _plot_parameters[9]; 	_linewidth 			= _plot_parameters[10]; 
	_voxel_width 				= _voxel_dimensions[0]; 	_voxel_height 	= _voxel_dimensions[1];	_voxel_thickness 	= _voxel_dimensions[2];
	
	_ROI_selection_diameter = _ROI_radius * 2;
	if(_ROI_selection_diameter % 2 == 1)
	{
		_profile_center_x = _ROI_x + 0.5; 
		_profile_center_y = _ROI_y + 0.5; 
	}
	else
	{
		_profile_center_x = _ROI_x; 
		_profile_center_y = _ROI_y; 
	}
	//run											("Specify...", "width=" + ROI_selection_diameter + " height=" + ROI_selection_diameter + " x=" + selection_center_x + " y=" + selection_center_y + " oval centered");							
	if(_profile_direction == x_direction)
	{
		_ROI_varying_direction 	= _profile_center_x;
		_ROI_constant_direction = _profile_center_y;
		_voxel_dimension	 	= _voxel_width;
	}
	else
	{
		_ROI_varying_direction 	= _profile_center_y;
		_ROI_constant_direction = _profile_center_x;
		_voxel_dimension 		= _voxel_height;
	}
	_data_shifted_distance		= _ROI_varying_direction - floor(_ROI_varying_direction);
	
	_profile_1st_edge			= floor((	_ROI_varying_direction - _profile_radius						) 	* _voxel_dimension	);
	_profile_2nd_edge			= ceil(( 	_ROI_varying_direction + _profile_radius 						) 	* _voxel_dimension	);		
	_ROI_1st_edge				= ( 		_ROI_varying_direction - _ROI_radius - _data_shifted_distance	) 	* _voxel_dimension;
	_ROI_2nd_edge				= ( 		_ROI_varying_direction + _ROI_radius - _data_shifted_distance	) 	* _voxel_dimension;		
	_center_coordinate 			= ( 		_ROI_varying_direction 											)	* _voxel_dimension;														
	
	if(_profile_direction == x_direction)
		makeLine				(_profile_1st_edge, _ROI_constant_direction, _profile_2nd_edge, _ROI_constant_direction );	
	else
		makeLine				(_ROI_constant_direction, _profile_1st_edge, _ROI_constant_direction, _profile_2nd_edge );		
	_profile 					= getProfile();
	_profile_elements			= _profile.length;
	getSelectionCoordinates		(_xpoints, _ypoints);
	_selection_length			= _xpoints[1] - _xpoints[0];
	_point_separation			= _selection_length / (_profile_elements - 1);
	_x_axis_coordinates 			= Array.slice(_sequential_values, _profile_1st_edge, _profile_2nd_edge + 1);
	_predicted_profile_coords 	= newArray(_profile_1st_edge, _ROI_1st_edge, _ROI_1st_edge, _ROI_1st_edge, _ROI_2nd_edge, _ROI_2nd_edge, _ROI_2nd_edge, _profile_2nd_edge);
	_predicted_profile 			= newArray(_base_value, _base_value, _base_value, _ref_value, _ref_value, _base_value, _base_value, _base_value );			
	//setResult					(1, ROI_number, profile[0]);
	for(i = 0; i < _profile.length; i++)							
		setResult				(i, _ROI_number, _profile[i]);
	_profile_extrema 			= set_plot_extrema(_profile, _predicted_profile, _tolerance, _lower_limits_scale, _upper_limits_scale, _difference_scale);
	_plot_ymin 					= _profile_extrema[0];
	_plot_ymax 					= _profile_extrema[1];
	_center_line_x 				= newArray(_center_coordinate, _center_coordinate);
	_center_line_y 				= newArray(_plot_ymin, _plot_ymax);
	if(_is_gradient_analysis)
	{
		if( _ref_value > _base_value)
			_predicted_profile 	= newArray(0.0, 0.0, _plot_ymax, 0.0, 0.0, _plot_ymin, 0.0, 0.0 );				
		else
			_predicted_profile 	= newArray(0.0, 0.0, _plot_ymin, 0.0, 0.0, _plot_ymax, 0.0, 0.0 );
	}
	Plot.create					(_title, _xlabel, _ylabel, _x_axis_coordinates, _profile																);
	Plot.setLimits				(_profile_1st_edge, _profile_2nd_edge, _plot_ymin, _plot_ymax														);
	set_plot_props				(_legend_entries, _legend_text_size, _x_frame_size, _y_frame_size, _text, _text_size, _text_xpos_ratio, _text_ypos_ratio, _add_line_color, _linewidth	);					
	Plot.setColor				("green"																										);
	Plot.add					("line", _predicted_profile_coords, _predicted_profile															);
	Plot.setColor				("yellow"																										);
	Plot.add					("line", _center_line_x, _center_line_y																			);					
	set_plot_data_props			(_data_line_color, _add_text_justification																		);			
	return _profile;
}
function listFiles(dir) 
{
	list 				= getFileList(dir);
	for (i = 0; i < list.length; i++) 
	{
	if(endsWith(list[i], "/"))
	   	listFiles	("------->" + dir + list[i]);
	else
		print		("------->" + (count++) + ": " + dir + list[i]);
     }
}
function listFolders(dir) 
{
	count 					= 1;
	list 					= getFileList(dir);
	folders					= newArray();
	for (i = 0; i < list.length; i++) 
    {
		if (File.isDirectory(dir + list[i]))
		{
	   		print			("------->" + (count++) + ": " + dir + list[i]);
			folders 		= Array.concat(folders, list[i] );
		}
     }
     return folders;
}
function make_directory_chain(top_directory, subdirectory_chain, folder_separator, make_top_directory, _print_directories_created)
{
	path 								= top_directory;
	subdirectory_chain_folders 			= split(subdirectory_chain, folder_separator);
	new_folder_paths 					= newArray(subdirectory_chain_folders.length);
	if(make_top_directory)
	{
		if(_print_directories_created)
			print							("------->Creating top directory:\n" + path );
		directory_created_successfully	= make_directory_recorded(path, DIRECTORIES_CREATED, _print_directories_created);
	}
	for(i = 0; i < subdirectory_chain_folders.length; i++)
	{
		path 							+= folder_separator + subdirectory_chain_folders[i];
		new_folder_paths[i] 			= path;
		if(_print_directories_created)
			print						("------->Creating directory:\n" + path );
		directory_created_successfully	= make_directory_recorded(path, DIRECTORIES_CREATED, _print_directories_created);
	}
	return new_folder_paths;											
}
function make_directory_parents(top_directory, bottom_directory, folder_separator, make_top_directory, _print_directories_created)
{
	bottom_directory_folders 			= split(bottom_directory, folder_separator);
	top_directory_folders 				= split(top_directory, folder_separator);
	return_statement					= "------->Top directory through target directory already exist\n";
	if(top_directory_folders.length != bottom_directory_folders.length)
	{
		if(make_top_directory)
		{
			if(!File.exists(top_directory))
			{
				directory_created_successfully	= make_directory_recorded(top_directory, DIRECTORIES_CREATED, _print_directories_created);		
				return_statement =  "------->Top directory created:\n" + top_directory;	
			}
			else
				return_statement = "";			
		}
		path 								= top_directory;
		return_statement = return_statement + "------->Subdirectories created:\n";
		subdirectories_2_create = Array.slice(bottom_directory_folders, top_directory_folders.length, bottom_directory_folders.length);		
		for(i = 0; i < subdirectories_2_create.length; i++)
		{
			path 							+= folder_separator + subdirectories_2_create[i];
			if(!File.exists(path))
				directory_created_successfully	= make_directory_recorded(path, DIRECTORIES_CREATED, _print_directories_created);		
			//else
			//	print("------->Directory already exists");
			return_statement = return_statement + path + "\n";
		}
		return subdirectories_2_create;
	}
	else
		return_statement = return_statement + "--------------->Top directory and target directory are the same\n";
	if(_print_directories_created)
		print(return_statement);
	return return_statement;											
}
function make_directory_recorded(_directory_2_create, _directory_creation_record, _print_directory_path)
{
	if(!File.exists(_directory_2_create))
	{
		if(_print_directory_path)
			print("\n------->Creating directory and adding it to the directory created log:\n" + _directory_2_create);
		File.makeDirectory							(_directory_2_create);
		_directory_creation_record					= Array.concat(_directory_creation_record, _directory_2_create);		
		if(!File.exists(_directory_2_create))
			print("**************ERROR: Directory creation failure*********************\n" + _directory_2_create);
		return false;
	}
	else
		return true;
	//else
	//	print("------->Directory already exists");
	//return File.exists							(directory_2_create);
}
function make_directory_set(_common_parent_directory, _subdirectory_set, _directory_creation_record, _print_directories_created)
{
	_all_subdirectories_created			= newArray();
	for(i = 0; i < _subdirectory_set.length; i++)
	{					
		_subdirectories_created 		= make_subdirectory(_common_parent_directory, _subdirectory_set[i], _directory_creation_record, _print_directories_created);
		_all_subdirectories_created 	= Array.concat(_all_subdirectories_created, _subdirectories_created);
	}
	return _all_subdirectories_created;
}
function make_subdirectory(_parent_directory, _subdirectory, _directory_creation_record, _print_directories_created)
{
	_parent_directory_folders 			= split(_parent_directory, File.separator);
	_subdirectory_folders 				= split(_subdirectory, File.separator);
	_directory_2_create 				= _parent_directory;
	_directories_created				= newArray();
	make_directory_recorded(_parent_directory, _directory_creation_record, _print_directories_created);
	for(i = 0; i < _subdirectory_folders.length; i++)
	{
		_directory_2_create = construct_valid_directory_path(_directory_2_create, _subdirectory_folders[i]);
		//_directory_2_create	+= _subdirectory_folders[i];
		make_directory_recorded(_directory_2_create, _directory_creation_record, false);
		_directories_created = Array.concat(_directories_created, _directory_2_create);
	}
	if(_print_directories_created)
	{
		print("-------> Subdirectories created:\n");
		Array.print(_directories_created);
	}
	return _directory_2_create;
}
function material_name_2_RSP(material_name, is_simulated_scan)
{
	material_names				= newArray("air", 	"PMP", 	"LDPE", 	"epoxy", "polystyrene", "PMMA", "acrylic", "delrin", "teflon"	);
	simulated_material_RSPs		= newArray(0.0013, 	0.877, 	0.9973, 	1.024,   1.0386,		1.144,  1.155,     1.356,     1.828		);
	experimental_material_RSPs	= newArray(0.0013, 	0.883, 	0.979,  	1.024,   1.024,			1.144,  1.160,     1.359,     1.79		); 	
	num_materials 				= material_names.length;
	if(is_simulated_scan)
		material_RSPs 			= simulated_material_RSPs;
	else
		material_RSPs 			= experimental_material_RSPs;
	for(j = 0; j < num_materials; j++)
		if( matches(material_name, material_names[j]) )
			return material_RSPs[j];
}
function modn(value, divisor)
{
	modulo_value 		= value;
	while(modulo_value >= divisor)
		modulo_value 	-= divisor;
	return modulo_value;
}
function opened_image_set_2_stack(_image_names_matching, _stack_title, _close_image_set)
{
	_open_image_list 		= getList("image.titles");	
	if(_close_image_set)
		run						("Images to Stack", "name=" + _stack_title + " title=" + _image_names_matching);			
	else
		run						("Images to Stack", "name=" + _stack_title + " title=" + _image_names_matching + " use keep");			
	Stack.getDimensions		(_width, _height, _channels, _slices, _frames);
	_stack_dimensions 		= newArray(_width, _height, _channels, _slices, _frames);
	return _stack_dimensions;			
}			
function open_image(image_type, directory, filename, print_path)
{
	image_path 					= construct_valid_file_path(directory, filename);	
	if(print_image_paths)
		print					("------->Opening image from:\n" + image_path);
	if(image_type == TXT)
		run						("Text Image... ", "open=" + image_path);
	if(image_type == PNG)
		open					(image_path);
	run							("Size...", "width=x_frame_size height=y_frame_size constrain average interpolation=Bilinear");	
	run							("Set Measurements...", "area mean standard min redirect = None decimal = 4");		
	INPUT_FILE_LIST				= Array.concat(INPUT_FILE_LIST, image_path);			
	return image_path;
}
function open_image_set(image_type, directory, filenames, print_image_paths)
{
	num_images_2_open 			= filenames.length;
	image_set 				= newArray(num_images_2_open);
	for(i = 0; i < num_images_2_open; i++)
	{
		image_filename 			= filenames[i] + image_type;
		image_set[i] 			= image_filename;
		open_image				(image_type, directory, image_filename, print_image_paths);
	}
	return image_set;
}			
function open_PNG_path(image_path, print_path)
{
	if(print_image_paths)
		print				("------->Opening PNG image from:\n" + image_path);
	open					(image_path);
	run						("Set Measurements...", "area mean standard min redirect = None decimal = 4");		
	run						("Size...", "width=x_frame_size height=y_frame_size constrain average interpolation=Bilinear");	
	INPUT_FILE_LIST			= Array.concat(INPUT_FILE_LIST, image_path);	
	return image_path;
}
function open_PNG_paths(PNG_paths, print_image_paths)
{
	num_PNGs_2_open 		= PNG_paths.length;
	image_set 				= newArray(num_PNGs_2_open);
	for(i = 0; i < num_PNGs_2_open; i++)
	{
		open_PNG_path		(PNG_paths[i], print_image_paths);		
		INPUT_FILE_LIST		= Array.concat(INPUT_FILE_LIST, PNG_paths[i]);			
	}
	//return image_set;
}			
function open_reconstructed_image(directory, filename)
{
	file_path 		= construct_valid_file_path(directory, filename);		
	run				("Text Image... ", "open=" + file_path);
	run				("Set Measurements...", "area mean standard min redirect = None decimal = 4");	
	INPUT_FILE_LIST	= Array.concat(INPUT_FILE_LIST, file_path);					
}																						
function order_comparison_data(unordered_data, ordering_indices, _num_ROIs_2_analyze, images_per_reconstruction, num_target_test_parameter_values)
{
	ordered_data 		= newArray(_num_ROIs_2_analyze * images_per_reconstruction * num_target_test_parameter_values);
	ordered_data_index 	= 0;
	for(i = 0; i < ordering_indices.length; i++)
		ordered_data[i] = unordered_data[ordering_indices[i]];
	return ordered_data;										
}
function parameter_indices_2_index(num_parameter_values_, parameter_indices_)
{
	index 			= 0;
	modulo_value 	= 1;
	for(n = 0; n < num_parameter_values_.length; n++)
	{
		index 		+= parameter_indices_[n] * modulo_value;
		modulo_value *= num_parameter_values_[n]; 
	}
	return index;
}
function parameter_string_2_index(_parameter_string_prefixes, _parameter_string) 
{
	_num_parameters = _parameter_string_prefixes.length;
	for(i = 0; i < _num_parameters; i++)
		if( _parameter_string_prefixes[i] == _parameter_string)
			return i;
	return -1;
}
function parameter_values_2_parameter_equals_short_strings(_parameter_values, _parameter_value_prefix, _parameter_value_string_precision)
{
	_num_parameter_values	= _parameter_values.length;
	_parameter_value_strings = newArray(_num_parameter_values);
	for(i = 0; i < _num_parameter_values; i++)
		_parameter_value_strings[i]	= _parameter_value_prefix + "=" + remove_trailing_zeros(d2s(_parameter_values[i], _parameter_value_string_precision), _parameter_value_string_precision);
	return _parameter_value_strings;
}
function parameter_values_2_parameter_equals_strings(_parameter_values, _parameter_value_prefix, _parameter_value_string_precision)
{
	_num_parameter_values	= _parameter_values.length;
	_parameter_value_strings = newArray(_num_parameter_values);
	for(i = 0; i < _num_parameter_values; i++)
		_parameter_value_strings[i]	= _parameter_value_prefix + "=" + d2s(_parameter_values[i], _parameter_value_string_precision);
	return _parameter_value_strings;
}
function parameter_value_test_folder_name_2_base(_folder_name, _target_test_parameter_prefix)
{
	_target_test_parameter_prefix_length 		= lengthOf		(_target_test_parameter_prefix);
	_start_index 		= indexOf		(_folder_name, _target_test_parameter_prefix );
	_end_index 			= _start_index + _target_test_parameter_prefix_length;
	_parameter_test_folder_basename 	= substring		(_folder_name, 0, _end_index);
	return _parameter_test_folder_basename;
}
function parameter_values_2_strings(_parameter_values, _parameter_value_string_precision)
{
	_num_parameter_values	= _parameter_values.length;
	_parameter_value_strings = newArray(_num_parameter_values);
	for(i = 0; i < _num_parameter_values; i++)
		_parameter_value_strings[i]	= d2s(_parameter_values[i], _parameter_value_string_precision);
	return _parameter_value_strings;
}
function path_2_phantom_name(path, reconstruction_data_folder)
{
	folder_names 					= split(path, File.separator);
	reconstruction_data_folder_name = substring(reconstruction_data_folder, 1);
	for(i = 0; i < folder_names.length; i++)
		if(folder_names[i] == reconstruction_data_folder_name)
			return folder_names[i + 1];
}
function PNG_sequence_2_animations(PNG_sequence_directory, PNG_sequence_paths, group_2_extract, num_groups, group_size, _property_Booleans, _PNG_basename, _PNG_image_Booleans, _animation_filename, _animated_GIF_info, _AVI_info, close_stack_after_saving)
{
	_print_input_PNG_paths			= _PNG_image_Booleans[0];
	_close_stack_images				= _PNG_image_Booleans[1];
	_print_multiplot_GIF_paths		= _animated_GIF_info[0];
	_close_multiplot_stack_images	= _animated_GIF_info[2];
	_GIF_frame_rate					= _animated_GIF_info[3];
	_print_multiplot_AVI_paths		= _AVI_info[0];
	_AVI_frame_rate					= _AVI_info[3];
	_AVI_compression_format			= _AVI_info[4];
	_write_GIF						= _property_Booleans[3];
	_overwrite_GIF					= _property_Booleans[4];
	_write_AVI						= _property_Booleans[5];
	_overwrite_AVI					= _property_Booleans[6];
	PNG_subsequence_paths 			= extract_subsequence_group(PNG_sequence_paths, group_2_extract, num_groups, group_size);	
	open_PNG_paths					(PNG_subsequence_paths, _print_input_PNG_paths);
	opened_image_set_2_stack		(_PNG_basename, _animation_filename, _close_stack_images);											
	stack_2_animated_GIF			(_animation_filename, PNG_sequence_directory, _animation_filename + GIF, _GIF_frame_rate, _write_GIF, _overwrite_GIF, _print_multiplot_GIF_paths, DONT_CLOSE_WINDOW);
	stack_2_AVI						(_animation_filename, PNG_sequence_directory, _animation_filename + AVI, _AVI_compression_format, _AVI_frame_rate, _write_AVI, _overwrite_AVI, _print_multiplot_AVI_paths, close_stack_after_saving);
	return PNG_subsequence_paths;
}
function PNG_sequence_2_stack(_PNG_sequence_paths, _group_2_extract, _num_groups, _group_size, _property_Booleans, _PNG_basename, _PNG_image_Booleans, _stack_title, _animated_GIF_info, _AVI_info, _close_stack_after_saving)
{
	_print_input_PNG_paths			= _PNG_image_Booleans[0];
	_close_stack_images				= _PNG_image_Booleans[1];
	_print_multiplot_GIF_paths		= _animated_GIF_info[0];
	_close_multiplot_stack_images	= _animated_GIF_info[2];
	_GIF_frame_rate					= _animated_GIF_info[3];
	_print_multiplot_AVI_paths		= _AVI_info[0];
	_AVI_frame_rate					= _AVI_info[3];
	_AVI_compression_format			= _AVI_info[4];
	_write_GIF						= _property_Booleans[3];
	_overwrite_GIF					= _property_Booleans[4];
	_write_AVI						= _property_Booleans[5];
	_overwrite_AVI					= _property_Booleans[6];
	_PNG_subsequence_paths 			= extract_subsequence_group(_PNG_sequence_paths, _group_2_extract, _num_groups, _group_size);	
	open_PNG_paths					(_PNG_subsequence_paths, _print_input_PNG_paths);
	opened_image_set_2_stack		(_PNG_basename, _stack_title, true);											
	//opened_image_set_2_stack		(_PNG_basename, _stack_title, _close_stack_images);											
	return _PNG_subsequence_paths;
}
function prefixes_2_filenames(common_basename, filename_prefixes)
{
	num_prefixes		= filename_prefixes.length;
	filenames			= newArray(num_prefixes);
	for(i = 0; i < num_prefixes; i++)
		filenames[i]	= filename_prefixes[i] + common_basename;
	return 				filenames;
}
function print_aligned_key_value_pairs(key, value, aligned_at_column)
{
	key_string_length = lengthOf(key);
	value_string_length = lengthOf(value);
	padding_spaces		= aligned_at_column - key_string_length;
	padding_string		= "";
	for(i = 0; i < padding_spaces; i++)
		padding_string += "-";
	print(key + padding_string + " = " + value);	
}
function print_by_iteration_data()
{
	print("-------> RSPs_by_iteration  		="	); 	Array.print(RSPs_by_iteration);
	print("-------> RSP_errors_by_iteration	="	); 	Array.print(RSP_errors_by_iteration);
	print("-------> std_devs_by_iteration  	="	); 	Array.print(std_devs_by_iteration);
}
function print_by_ROI_data()
{
	print("-------> RSPs_by_ROI  		="	); 	Array.print(RSPs_by_ROI);
	print("-------> RSP_errors_by_ROI  	="	); 	Array.print(RSP_errors_by_ROI);
	print("-------> std_devs_by_ROI  	="	); 	Array.print(std_devs_by_ROI);
}		
function print_compared_data_subdirectory_info()
{
	print_section_separator("Printing compared data subdirectory info", PRINT_MAJOR_SECTION													);
	print("-------> multiplot_compared_data_subdirectory 						=\n " 	+ multiplot_compared_data_subdirectory				);	
	print("-------> multiplot_compared_RSP_data_subdirectory 					=\n " 	+ multiplot_compared_RSP_data_subdirectory			);		
	print("-------> multiplot_compared_RSP_error_data_subdirectory 				=\n " 	+ multiplot_compared_RSP_error_data_subdirectory	);	
	print("-------> multiplot_compared_std_dev_data_subdirectory 				=\n " 	+ multiplot_compared_std_dev_data_subdirectory		);	
	print("-------> multiplot_compared_TV_data_subdirectory 					=\n " 	+ multiplot_compared_TV_data_subdirectory			);		
}
function print_section_separator(section_heading, print_major_section)
{
	slash_section_separator 			= "///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////";
	asterisk_section_separator 			= "//*************************************************************************************************************************************************************************************************************************************************************************************************";
	dash_section_separator 				= "//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
	length_slash_section_separator 		= lengthOf(slash_section_separator);
	length_asterisk_section_separator 	= lengthOf(asterisk_section_separator);
	length_dash_section_separator 		= lengthOf(dash_section_separator);
	length_section_heading 				= lengthOf(section_heading);
	asterisk_text_ratio 				= 332/272;
	asterisk_length_diff 				= length_asterisk_section_separator - length_section_heading*asterisk_text_ratio;
	dash_length_diff 					= length_asterisk_section_separator - length_section_heading*asterisk_text_ratio;
	length_asterisk_heading_prefix 		= floor(asterisk_length_diff / 2);
	length_asterisk_heading_suffix 		= ceil(asterisk_length_diff / 2);
	length_dash_heading_prefix 			= floor(dash_length_diff / 2);
	length_dash_heading_suffix 			= ceil(dash_length_diff / 2);
	dash_heading_suffix 				= " ";
	dash_heading_prefix 				= "//";
	asterisk_heading_suffix 			= " ";
	asterisk_heading_prefix 			= "//";
	for(i = 0; i < length_asterisk_heading_prefix - 3; i++)
		asterisk_heading_prefix 		+= "*";
	for(i = 0; i < length_asterisk_heading_suffix - 3; i++)
		asterisk_heading_suffix 		+= "*";
	for(i = 0; i < length_dash_heading_prefix - 3; i++)
		dash_heading_prefix 			+= "-";
	for(i = 0; i < length_dash_heading_suffix - 3; i++)
		dash_heading_suffix 			+= "-";
	asterisk_heading_prefix	 			+= " ";
	asterisk_heading_suffix 			+= "";
	asterisk_heading_separator 			= asterisk_heading_prefix + section_heading + asterisk_heading_suffix;
	dash_heading_prefix 				+= " ";
	dash_heading_suffix 				+= "";
	dash_heading_separator 				= dash_heading_prefix + section_heading + dash_heading_suffix;
	length_asterisk_heading_prefix 		= lengthOf(asterisk_heading_prefix);
	length_asterisk_heading_suffix 		= lengthOf(asterisk_heading_suffix);
	length_asterisk_heading_separator 	= lengthOf(asterisk_heading_separator);
	length_dash_heading_prefix 			= lengthOf(dash_heading_prefix);
	length_dash_heading_suffix 			= lengthOf(dash_heading_suffix);
	length_dash_heading_separator 		= lengthOf(dash_heading_separator);
	
	asterisk_heading_addition 			= 150;
	dash_text_ratio 					= 1;
	dash_heading_addition				= asterisk_heading_addition * dash_text_ratio;//length_dash_heading_separator/length_asterisk_heading_separator;
	slash_section_separator_addition	= asterisk_heading_addition * asterisk_text_ratio;//length_slash_section_separator/length_asterisk_heading_separator;
	
	for(i = 0; i < asterisk_heading_addition; i++)
	{
		asterisk_heading_separator += "*";
		asterisk_section_separator += "*";
	}
	for(i = 0; i < dash_heading_addition; i++)
	{
		dash_section_separator += "-";
		dash_heading_separator += "-";
	}
	for(i = 0; i < slash_section_separator_addition; i++)
		slash_section_separator += "/";
	
	//asterisk_heading_separator += "//";
	slash_section_separator += "//";
	asterisk_section_separator += "//";
	dash_section_separator += "//";
	asterisk_heading_separator += "//";
	dash_heading_separator += "//";
	if(print_major_section)
	{
		print							(slash_section_separator);
		print							(asterisk_section_separator);
		print							(asterisk_heading_separator);
		print							(asterisk_section_separator);
		print							(slash_section_separator);
	}
	else
	{
		print							(dash_section_separator);
		print							(dash_heading_separator);
		print							(dash_section_separator);			
	}
}
function print_current_parameter_value_test_info()
{
	print_section_separator("Printing current parameter value test info", PRINT_MAJOR_SECTION											);
	print("-------> current_test_start_folder 								=\n " 	+ current_test_start_folder							);				
	print("-------> current_test_end_folder 						 		=\n " 	+ current_test_end_folder							);				
	print("-------> current_test_folder_basename							=\n " 	+ current_test_folder_basename						);
	print("-------> current_test_folders									: 	" 														);	Array.print(current_test_folders);			
	print("-------> current_test_parameter_value_strings 		 			: 	" 														);	Array.print(current_test_parameter_value_strings);		
	print("-------> current_test_parameter_values_string			 		=\n " 	+ current_test_parameter_values_string				);				
	print("-------> current_test_plot_title_parameter_values_string  		=\n " 	+ current_test_plot_title_parameter_values_string	);						
	print("-------> current_test_parameter_equals_strings  					: 	"														);	Array.print(current_test_parameter_equals_strings);					
	print("-------> current_test_parameter_equals_short_strings  			: 	"														);	Array.print(current_test_parameter_equals_short_strings);									
	print("-------> current_test_file_suffix  								=\n " 	+ current_test_file_suffix							);									
}
function print_current_parameter_value_test_multiplot_info()
{
	print_section_separator("Printing current parameter value test multiplot info...", PRINT_MAJOR_SECTION									);
	print("-------> current_test_multiplot_start_folder 			 		=\n " 	+ current_test_multiplot_start_folder					);	
	print("-------> current_test_multiplot_end_folder 			 			=\n " 	+ current_test_multiplot_end_folder						);				
	print("-------> current_test_multiplot_folders					 		= 	" 															);	Array.print(current_test_multiplot_folders);
	print("-------> current_test_multiplot_value_strings 		 			= 	" 															);	Array.print(current_test_multiplot_value_strings);	
	print("-------> current_test_multiplot_title_parameter_values_string	=\n " 	+ current_test_multiplot_title_parameter_values_string	);								
	print("-------> current_test_multiplot_parameter_values_string  		=\n " 	+ current_test_multiplot_parameter_values_string		);						
	print("-------> current_test_multiplot_parameter_equals_strings 		: 	" 															);	Array.print(current_test_multiplot_parameter_equals_strings);	
	print("-------> current_test_multiplot_parameter_equals_short_strings 	: 	" 															);	Array.print(current_test_multiplot_parameter_equals_short_strings);	
	print("-------> current_test_multiplot_file_suffix  					=\n " 	+ current_test_multiplot_file_suffix					);									
}
function print_current_parameter_value_test_parent_directory_info()
{	
	print_section_separator("Printing current parameter test input/output directories, paths, and filenames", true																					);
	print		("------->\n parameter_value_test_identifier												=\n" 	+ parameter_value_test_identifier												); 					
	print		("------->\n parameter_value_tests_input_directory_parent 									=\n" 	+ parameter_value_tests_input_directory_parent									);
	print		("------->\n parameter_value_tests_output_directory_parent 									=\n"	+ parameter_value_tests_output_directory_parent									);
	print		("------->\n current_parameter_value_test_output_directory_parent 							=\n"	+ current_parameter_value_test_output_directory_parent							);	
	print		("------->\n current_parameter_value_test_parent_folder 									=\n"	+ current_parameter_value_test_parent_folder									);	
	print		("------->\n current_parameter_value_test_parent_directory 									=\n"	+ current_parameter_value_test_parent_directory									);	
	print		("------->\n current_parameter_value_test_plot_parent_folder 								=\n"	+ current_parameter_value_test_plot_parent_folder								);	
	print		("------->\n current_parameter_value_test_plot_parent_directory 							=\n"	+ current_parameter_value_test_plot_parent_directory							);	
	print		("------->\n current_parameter_value_test_multiplot_parent_folder 							=\n"	+ current_parameter_value_test_multiplot_parent_folder							);	
	print		("------->\n current_parameter_value_test_multiplot_parent_directory 						=\n"	+ current_parameter_value_test_multiplot_parent_directory						);	
					
	print		("------->\n current_parameter_value_test_output_directory_path 							=\n"	+ current_parameter_value_test_output_directory_path							);	
	print		("------->\n current_parameter_value_test_subdirectory 										=\n"	+ current_parameter_value_test_subdirectory										);	
						
	print		("------->\n current_parameter_value_test_TV_analysis_subdirectory 							=\n"	+ current_parameter_value_test_TV_analysis_subdirectory							);	
	print		("------->\n current_parameter_value_test_TV_analysis_parameter_value_subdirectory 			=\n"	+ current_parameter_value_test_TV_analysis_parameter_value_subdirectory			);	
	print		("------->\n current_parameter_value_test_ROI_analysis_subdirectory 						=\n"	+ current_parameter_value_test_ROI_analysis_subdirectory						);	
	print		("------->\n current_parameter_value_test_ROI_analysis_short_subdirectory 					=\n"	+ current_parameter_value_test_ROI_analysis_short_subdirectory					);	
	print		("------->\n current_parameter_value_test_ROI_analysis_parameter_value_subdirectory 		=\n"	+ current_parameter_value_test_ROI_analysis_parameter_value_subdirectory		);	
	print		("------->\n current_parameter_value_test_ROI_analysis_parameter_value_short_subdirectory	=\n"	+ current_parameter_value_test_ROI_analysis_parameter_value_short_subdirectory	);	
								
	print		("------->\n current_parameter_value_test_TV_analysis_path  								=\n"	+ current_parameter_value_test_TV_analysis_path									);	
	print		("------->\n current_parameter_value_test_TV_analysis_parameter_value_path  				=\n"	+ current_parameter_value_test_TV_analysis_parameter_value_path					);	
	print		("------->\n current_parameter_value_test_ROI_analysis_path  								=\n"	+ current_parameter_value_test_ROI_analysis_path								);	
	print		("------->\n current_parameter_value_test_ROI_analysis_parameter_value_path  				=\n"	+ current_parameter_value_test_ROI_analysis_parameter_value_path				);	
						
	print		("------->\n current_parameter_value_test_plot_ROI_analysis_path  							=\n"	+ current_parameter_value_test_plot_ROI_analysis_path							);	
	print		("------->\n current_parameter_value_test_plot_TV_analysis_path  							=\n"	+ current_parameter_value_test_plot_TV_analysis_path							);	
						
	print		("------->\n current_parameter_value_test_multiplot_ROI_analysis_path  						=\n"	+ current_parameter_value_test_multiplot_ROI_analysis_path						);	
	print		("------->\n current_parameter_value_test_multiplot_ROI_analysis_parameter_value_path  		=\n"	+ current_parameter_value_test_multiplot_ROI_analysis_parameter_value_path		);	
	print		("------->\n current_parameter_value_test_multiplot_TV_analysis_path 						=\n"	+ current_parameter_value_test_multiplot_TV_analysis_path						);	
	print		("------->\n current_parameter_value_test_multiplot_TV_analysis_parameter_value_path 		=\n"	+ current_parameter_value_test_multiplot_TV_analysis_parameter_value_path		);												
}
function print_ImageJ_info()
{
	print_section_separator("Printing ImageJ info", PRINT_MAJOR_SECTION);
	print("-------> ImageJ_plugins_directory =\n ", ImageJ_plugins_directory);
	print("-------> ImageJ_macro_directory =\n ", ImageJ_macro_directory);
	print("-------> ImageJ_launch_directory =\n ", ImageJ_launch_directory);
	print("-------> ImageJ_previous_macro =\n ", ImageJ_previous_macro);
	print("-------> github_macro_directory =\n ", github_macro_directory);
	print("-------> reconstruction_data_directory_C =\n ", reconstruction_data_directory_C);
	print("-------> reconstruction_data_directory_D =\n ", reconstruction_data_directory_D);
	print("-------> ROI_analysis_macro_filename =\n ", ROI_analysis_macro_filename);
	print("-------> ROI_analysis_macro_path =\n ", ROI_analysis_macro_path);
	print("-------> multiplotting_macro_filename =\n ", multiplotting_macro_filename);
	print("-------> multiplotting_macro_path =\n ", multiplotting_macro_path);
	print("-------> test_result_comparison_macro_filename =\n ", test_result_comparison_macro_filename);
	print("-------> test_result_comparison_macro_path =\n ", test_result_comparison_macro_path);
	print("-------> auto_break_filename =\n ", auto_break_filename);
	print("-------> auto_break_path =\n ", auto_break_path);		
}
function print_multiplot_parameter_info()
{			
	print_section_separator("Printing parameter test multiplot info", PRINT_MAJOR_SECTION								);
	print		("-------> multiplot_parameter_index 					=\n " 	+ multiplot_parameter_index				);
	print		("-------> multiplot_parameter	 						=\n " 	+ multiplot_parameter					);
	print		("-------> multiplot_parameter_prefix 					=\n " 	+ multiplot_parameter_prefix			);
	print		("-------> multiplot_parameter_values 					: 	" 											);	Array.print	(multiplot_parameter_values);
	print		("-------> multiplot_parameter_value_strings 			: 	" 											);	Array.print	(multiplot_parameter_value_strings);
	print		("-------> multiplot_parameter_value_short_strings 		: 	" 											);	Array.print	(multiplot_parameter_value_short_strings);
	print		("-------> multiplot_parameter_equals_strings 			: 	" 											);	Array.print	(multiplot_parameter_equals_strings);	
	print		("-------> multiplot_parameter_equals_short_strings 	: 	" 											);	Array.print	(multiplot_parameter_equals_short_strings);	
	print		("-------> multiplot_test_parameter_min_value	 		=\n " 	+ multiplot_test_parameter_min_value	);
	print		("-------> multiplot_test_parameter_max_value	 		=\n " 	+ multiplot_test_parameter_max_value	);
	print		("-------> multiplot_parameter_min_value_string	 		=\n " 	+ multiplot_parameter_min_value_string	);
	print		("-------> multiplot_parameter_max_value_string	 		=\n " 	+ multiplot_parameter_max_value_string	);		
	print		("-------> multiplot_parameter_range_suffix	 			=\n " 	+ multiplot_parameter_range_suffix		);
	print		("-------> multiplot_parameter_range_filenaming	 		=\n " 	+ multiplot_parameter_range_filenaming	);
	print		("-------> multiplot_parameter_range_plots	 			=\n " 	+ multiplot_parameter_range_plots		);
	print		("-------> num_multiplot_parameter_values 				=\n " 	+ num_multiplot_parameter_values		);
	print		("-------> reduced_parameter_values_lower 				: 	" 											);	Array.print	(reduced_parameter_values_lower);
	print		("-------> reduced_parameter_values_upper 				: 	" 											);	Array.print	(reduced_parameter_values_upper);
	print		("-------> reduced_parameter_values 					: 	" 											);	Array.print	(reduced_parameter_values);
	print		("-------> reduced_parameter_value_offsets 				: 	"											);	Array.print	(reduced_parameter_value_offsets);
	print		("-------> reduced_modulo_values     					: 	"											);	Array.print	(reduced_modulo_values);
	print		("-------> reduced_target_test_parameter_index 			=\n " 	+ reduced_target_test_parameter_index	);	
	print		("-------> reduced_total_combinations 					=\n " 	+ reduced_total_combinations			);
	print		("-------> all_reduced_parameter_combination_indices	: 	"											);	Array.print	(all_reduced_parameter_combination_indices);
	print		("-------> all_reduced_parameter_combinations 			: 	"											);	Array.print	(all_reduced_parameter_combinations);
	print		("-------> all_reduced_folder_strings 					: 	"											);	Array.print	(all_reduced_folder_strings);
	print		("-------> all_reduced_path_strings   					: 	"											);	Array.print	(all_reduced_path_strings);				
}
function print_input_output_filenames()
{
	print_section_separator("Printing input/output filenames of ROI analysis, parameter value test comparisons, and parameter value test comparison multiplots", PRINT_MAJOR_SECTION);
	print		("\n*ROI analysis input/output filenames----------------------------> "															);	
	print		("-------> ROI_analysis_TV_input_filename 							=\n " 	+ ROI_analysis_TV_input_filename					);									// 
	print		("-------> ROI_analysis_RSP_output_filename 						=\n " 	+ ROI_analysis_RSP_output_filename					);									// 
	print		("-------> ROI_analysis_RSP_error_output_filename 					=\n " 	+ ROI_analysis_RSP_error_output_filename			);									// 
	print		("-------> ROI_analysis_std_dev_output_filename 					=\n " 	+ ROI_analysis_std_dev_output_filename				);									// 
	print		("-------> ROI_analysis_TV_output_filename 							=\n " 	+ ROI_analysis_TV_output_filename					);									// 
	
	print		("\n*Parameter value test input/output filenames--------------------> "															);	
	print		("-------> parameter_value_test_RSP_input_filename 					=\n " 	+ parameter_value_test_RSP_input_filename			);									// 
	print		("-------> parameter_value_test_RSP_error_input_filename 			=\n " 	+ parameter_value_test_RSP_error_input_filename		);									// 
	print		("-------> parameter_value_test_std_dev_input_filename 				=\n " 	+ parameter_value_test_std_dev_input_filename		);									// 
	print		("-------> parameter_value_test_TV_input_filename 					=\n " 	+ parameter_value_test_TV_input_filename			);									// 
	print		("-------> parameter_value_test_RSP_output_filename 				=\n " 	+ parameter_value_test_RSP_output_filename			);									// 
	print		("-------> parameter_value_test_RSP_error_output_filename 			=\n " 	+ parameter_value_test_RSP_error_output_filename	);									// 
	print		("-------> parameter_value_test_std_dev_output_filename 			=\n " 	+ parameter_value_test_std_dev_output_filename		);									// 
	print		("-------> parameter_value_test_TV_output_filename 					=\n " 	+ parameter_value_test_TV_output_filename			);									// 
	
	print		("\n*Multiplot input/output filenames-------------------------------> "															);	
	print		("-------> multiplot_RSP_input_filename 							=\n " 	+ multiplot_RSP_input_filename						);									// 
	print		("-------> multiplot_RSP_error_input_filename 						=\n " 	+ multiplot_RSP_error_input_filename				);									// 
	print		("-------> multiplot_std_dev_input_filename 						=\n " 	+ multiplot_std_dev_input_filename					);									// 
	print		("-------> multiplot_TV_input_filename 								=\n " 	+ multiplot_TV_input_filename						);									// 
	print		("-------> multiplot_RSP_output_basename 							=\n " 	+ multiplot_RSP_output_basename						);									// 
	print		("-------> multiplot_RSP_error_output_basename 						=\n " 	+ multiplot_RSP_error_output_basename				);									// 
	print		("-------> multiplot_std_dev_output_basename 						=\n " 	+ multiplot_std_dev_output_basename					);									// 
	print		("-------> multiplot_TV_output_basename 							=\n " 	+ multiplot_TV_output_basename						);									// 		
}
function print_test_parameter_file_info()
{	
	print_section_separator("Printing parameter value test info...", PRINT_MAJOR_SECTION											);
	print		("-------> parameter_test_number							=\n" 	+ parameter_test_number							);
	print		("-------> parameter_test_info_filename						=\n" 	+ parameter_test_info_filename					);
	print		("-------> num_target_test_parameter_tests					=\n"	+ num_target_test_parameter_tests				);
	print		("-------> num_parameters									=\n"	+ num_parameters								);	
	print		("-------> parameter_values									: "														);	Array.print	(parameter_values);
	print		("-------> num_parameter_values								: "														);	Array.print	(num_parameter_values);
	print		("-------> parameter_value_offsets 							: "														);	Array.print	(parameter_value_offsets);
	print		("-------> parameter_string_prefixes						: "														);	Array.print	(parameter_string_prefixes);
	print		("-------> parameter_string_short_prefixes					: "														);	Array.print	(parameter_string_short_prefixes);		
	print		("-------> parameter_string_precisions						: "														);	Array.print	(parameter_string_precisions);
	print		("-------> modulo_values 									: "														);	Array.print	(modulo_values);
	
}
function print_target_test_parameter_info()
{	
	print_section_separator("Printing parameter value test info...", PRINT_MAJOR_SECTION											);
	print		("-------> target_test_parameter							=\n" 	+ target_test_parameter							);	
	print		("-------> target_test_parameter_index						=\n" 	+ target_test_parameter_index					);	
	print		("-------> target_test_parameter_prefix						=\n" 	+ target_test_parameter_prefix					);	
	print		("-------> target_test_parameter_values 					: "														);	Array.print(target_test_parameter_values);	
	print		("-------> target_test_parameter_value_strings 				: "														);	Array.print(target_test_parameter_value_strings);
	print		("-------> target_test_parameter_value_short_strings 		: "														);	Array.print(target_test_parameter_value_short_strings);
	print		("-------> target_test_parameter_equals_strings 			: "														);	Array.print(target_test_parameter_equals_strings);	
	print		("-------> target_test_parameter_equals_short_strings 		: "														);	Array.print(target_test_parameter_equals_short_strings);	
	print		("-------> target_test_parameter_min_value					=\n" 	+ target_test_parameter_min_value				);		
	print		("-------> target_test_parameter_max_value					=\n" 	+ target_test_parameter_max_value				);		
	print		("-------> target_test_parameter_min_value_string			=\n" 	+ target_test_parameter_min_value_string		);		
	print		("-------> target_test_parameter_max_value_string			=\n" 	+ target_test_parameter_max_value_string		);			
	print		("-------> target_test_parameter_range_suffix				=\n" 	+ target_test_parameter_range_suffix			);		
	print		("-------> target_test_parameter_range_filenaming			=\n" 	+ target_test_parameter_range_filenaming		);		
	print		("-------> target_test_parameter_range_plots				=\n" 	+ target_test_parameter_range_plots				);		
	print		("-------> num_target_test_parameter_values 				=\n"	+ num_target_test_parameter_values				);
	print		("-------> total_combinations 								=\n" 	+ total_combinations							);
	print		("-------> all_parameter_combination_indices				: "														);	Array.print	(all_parameter_combination_indices);
	print		("-------> all_parameter_combinations						: "														);	Array.print	(all_parameter_combinations);
	print		("-------> all_folder_strings								: "														);	Array.print	(all_folder_strings);
	print		("-------> all_path_strings									: "														);	Array.print	(all_path_strings);
	}
function print_reconstructed_image_info()
{	
	print_section_separator("Printing reconstructed image(s) properties", PRINT_MAJOR_SECTION									);
	print		("\nReconstructed image(s) properties-------------------> "														);	
	print		("-------> voxel_dimensions								: "														);	Array.print	(voxel_dimensions);	
	print		("-------> voxel_width 									=\n " 	+ voxel_width									);									// 
	print		("-------> voxel_height									=\n " 	+ voxel_height									);
	print		("-------> voxel_thickness								=\n " 	+ voxel_thickness								);
	//print		("-------> voxel_dimensions								=\n " 	+ voxel_dimensions								);
	print		("-------> voxels_per_mm								=\n " 	+ voxels_per_mm									);
	print		("-------> num_recon_iterations							=\n " 	+ num_recon_iterations							);
	print		("-------> images_per_reconstruction					=\n " 	+ images_per_reconstruction						);
	print		("-------> reconstructed_image_strings					: "														);	Array.print	(reconstructed_image_strings);
	print		("-------> reconstructed_image_folders					: "														);	Array.print	(reconstructed_image_folders);
}
function print_ROI_analysis_info()
{	
	print_section_separator("Printing reconstructed image(s)/ROI analysis configurations/parameters", PRINT_MAJOR_SECTION		);
	print		("\nImage analysis configuration/parameters-------------> "														);	
	print		("-------> first_iteration_2_analyze					=\n " 	+ first_iteration_2_analyze						);
	print		("-------> last_iteration_2_analyze						=\n " 	+ last_iteration_2_analyze						);
	print		("-------> num_iterations_2_analyze						=\n " 	+ num_iterations_2_analyze						);
	print		("-------> iterations_2_analyze							: "														);	Array.print	(iterations_2_analyze);	
	print		("-------> iterations_2_analyze_string_precision		=\n " 	+ iterations_2_analyze_string_precision			);
	print		("-------> iterations_2_analyze_strings					: "														);	Array.print	(iterations_2_analyze_strings);
	print		("-------> iterations_2_analyze_folders					: "														);	Array.print	(iterations_2_analyze_folders);
	print		("-------> first_slice_2_analyze						=\n " 	+ first_iteration_2_analyze						);
	print		("-------> last_slice_2_analyze							=\n " 	+ last_iteration_2_analyze						);
	print		("-------> num_slices_2_analyze							=\n " 	+ num_slices_2_analyze							);
	print		("-------> slices_2_analyze								: "														);	Array.print	(slices_2_analyze);
	print		("-------> slices_2_analyze_string_precision			=\n " 	+ slices_2_analyze_string_precision				);
	print		("-------> slices_2_analyze_strings						: "														);	Array.print	(slices_2_analyze_strings);
	print		("-------> slices_2_analyze_folders						: "														);	Array.print	(slices_2_analyze_folders);
	print		("-------> num_ROIs_2_analyze							=\n " 	+ num_ROIs_2_analyze							);
	print		("-------> num_ROI_selection_diameters					=\n " 	+ num_ROI_selection_diameters					);
	print		("-------> ROI_selection_diameters						: "														);	Array.print	(ROI_selection_diameters);
	print		("-------> ROI_selection_diameter_string_precision		=\n " 	+ ROI_selection_diameter_string_precision		);
	print		("-------> ROI_selection_diameter_strings				: "														);	Array.print	(ROI_selection_diameter_strings);
	print		("-------> ROI_selection_diameter_folders				: "														);	Array.print	(ROI_selection_diameter_folders);	
}
function print_ROI_definitions()
{
	print_section_separator("Printing ROI definitions and info", PRINT_MAJOR_SECTION				);
	print	("-------> phantom_basename					= " 	+ phantom_basename					);
	print	("-------> ROI_definitions_filename			= " 	+ ROI_definitions_filename			);
	print	("-------> ROI_definitions_file_path		= " 	+ ROI_definitions_file_path			);
	print	("-------> num_ROIs_2_analyze 				= " 	+ num_ROIs_2_analyze				);
	print	("-------> ROI_material_names 				= "											); 	Array.print(ROI_material_names);	
	print	("-------> ROI_material_RSPs 				= "											); 	Array.print(ROI_material_RSPs);	
	print	("-------> ROI_labels 						= "											); 	Array.print(ROI_labels);	
	print	("-------> ROI_label_nicknames 				= "											); 	Array.print(ROI_label_nicknames);	
	print	("-------> ROI_shapes 						= "											); 	Array.print(ROI_shapes);	
	print	("-------> ROI_diameters 					= "											); 	Array.print(ROI_diameters);	
	print	("-------> ROI_selection_radii 				= "											); 	Array.print(ROI_selection_radii);	
	print	("-------> ROI_profile_radius_beyond_ROI	= " 	+ ROI_profile_radius_beyond_ROI		);
	print	("-------> bulk_material 					= " 	+ bulk_material						);
	print	("-------> bulk_material_RSP 				= " 	+ bulk_material_RSP					);
}		
function prompt_4_image_set(directory, image_type, stack_suffix, dialog_position_x, dialog_position_y, dialog_character_width)
{		
	directory_folder 			= File.getName(directory_path);
	default_image_basename		= "x_";
	Dialog.create				("Specify reconstructed image set to open");
	Dialog.addString			("Basename common to images in set", default_image_basename); 
	Dialog.addNumber			("Specify 1st image #", 0, 0, dialog_character_width, ""); 
	Dialog.addNumber			("Specify 2nd image #", 12, 0, dialog_character_width, ""); 
	Dialog.addChoice			("Combine image set into image stack", newArray(TRUE, FALSE), TRUE);
	Dialog.setLocation			(dialog_position_x, dialog_position_y);
	Dialog.show					();
	_image_basename 				= Dialog.getString();
	first_image_number 			= Dialog.getNumber();
	second_image_number 		= Dialog.getNumber();
	do_image_set_2_stack	 	= Dialog.getChoice();
	filenames					= number_range_2_filenames(_image_basename, "", first_image_number, second_image_number, 0);
	image_set					= open_image_set(image_type, directory, filenames, TRUE);
	stack_name					= _image_basename + stack_suffix;
	if(do_image_set_2_stack == TRUE)
		opened_image_set_2_stack(_image_basename, stack_name, "false");
	return 						stack_name;
}
function pt()
{
	print("=======>Debugging location marker print #" + debug_print_counter++);
}	
function query_image_dimensions(directory_path, initial_iterate_filename)
{
	file_path 			= construct_valid_file_path(directory_path, initial_iterate_filename);		
	run					("Text Image... ", "open=" + file_path);
	getDimensions		(image_columns, image_height, image_channels, image_layers, image_frames);
	image_rows 			= image_columns;
	image_slices 		= image_height / image_rows;
	image_dimensions 	= newArray(image_columns, image_rows, image_slices, image_height, image_channels, image_layers, image_frames);
	run					("Close");	
	return image_dimensions;
}
function query_open_image_dimensions()
{
	getDimensions		(image_columns, image_height, image_channels, image_layers, image_frames);
	image_rows 			= image_columns;
	image_slices 		= image_height / image_rows;
	image_dimensions 	= newArray(image_columns, image_rows, image_slices, image_height, image_channels, image_layers, image_frames);
	//run				("Close");	
	return image_dimensions;
}
function read_ROI_definitions(_path, _reconstruction_data_folder, _ROI_definition_file_directory, _ROI_definitions_filename_suffix, _ROI_parameter_value_parseFloat, _print_path)
{
	_phantom_name 								= path_2_phantom_name(_path, _reconstruction_data_folder);
	_ROI_definition_filename					= _phantom_name_folder + _ROI_definitions_filename_suffix;
	_ROI_definition_file_key_value_pairs		= file_2_array(_ROI_definition_file_directory, _ROI_definition_filename, _print_path);
	_num_ROI_key_value_pairs					= _ROI_definition_file_key_value_pairs.length;
	_ROI_definition_file_key_values				= newArray(_num_ROI_key_value_pairs);
	_ROI_parameter_keys 						= newArray(_num_ROI_key_value_pairs);
	_bulk_material_decoding_op					= "none";//	= epoxy
	_ROI_profile_radius_beyond_ROI_decoding_op 	= "parseFloat";//	= 8														
	_ROI_selection_radii_decoding_op			= "parseFloat"; //	= 3.5, 4, 6
	_ROI_shapes_decoding_op						= "none";// = circular_ROI, circular_ROI, circular_ROI, circular_ROI, circular_ROI, circular_ROI, circular_ROI, circular_ROI
	_ROI_material_names_decoding_op				= "none"; //= air, air, PMP, LDPE,	polystyrene, acrylic, delrin, teflon
	_ROI_labels_decoding_op						= "none"; //= Air(bottom), Air(top), PMP, LDPE, Polystyrene, Acrylic, Delrin, Teflon
	_ROI_diameters_decoding_op					= "parseFloat";	//= 12.2, 12.2, 12.2, 12.2, 12.2, 12.2, 12.2, 12.2
	_ROI_parameter_decoding_ops					= newArray(_bulk_material_decoding_op, _ROI_profile_radius_beyond_ROI_decoding_op, _ROI_selection_radii_decoding_op, _ROI_shapes_decoding_op, _ROI_material_names_decoding_op, _ROI_labels_decoding_op, _ROI_diameters_decoding_op);	
	ROI_definition_numbers_decoding_op			= "parseFloat";	
	ROI_definition_strings_decoding_op			= "none";		
	_all_ROI_parameter_values 					= newArray();
	for(i = 0; i < num_ROI_key_value_pairs; i++)
	{
		_ROI_key_value_pair_separation_index	= indexOf(_ROI_definition_file_key_value_pairs[i], "=");
		_ROI_values_string						= substring(_ROI_definition_file_key_value_pairs[i], _ROI_key_value_pair_separation_index + 1);
		_ROI_parameter_key 						= substring(_ROI_definition_file_key_value_pairs[i], 0, _ROI_key_value_pair_separation_index);
		_spaceless_key_string 					= strip_surrounding_spaces(_ROI_parameter_key);
		_ROI_parameter_keys[i] 					= _spaceless_key_string;
		_ROI_parameter_values					= split(_ROI_values_string, ",");
		_num_ROI_parameter_values 				= _ROI_parameter_values.length;
		for(j = 0; j < _num_ROI_parameter_values; j++)
		{
			_spaceless_value_string 			= strip_surrounding_spaces(_ROI_parameter_values[j]);
			_ROI_parameter_values[j] 			= _spaceless_value_string;
			if(_ROI_parameter_decoding_ops[i] == _ROI_parameter_value_parseFloat)
				_ROI_parameter_values[j] 		= parseFloat(_spaceless_value_string);
		}
		_all_ROI_parameter_values 				= Array.concat(_all_ROI_parameter_values, _num_ROI_parameter_values, _ROI_parameter_values);
	}
	return _all_ROI_parameter_values;	
}
function remove_N_trailing_characters(_string, _N)
{
	return substring(_string, 0, lengthOf(_string) - _N);	
}
function remove_trailing_zeros(value_string, max_precision)
{
	min_precision 			= 0;
	value 					= parseFloat(value_string);
	for(i = 0; i < max_precision; i++)
	{
		new_value_string 	= d2s(value, i);
		new_value 			= parseFloat(new_value_string);
		if(value - new_value == 0)
			return new_value_string;
	}
	return value_string;
}
function remove_trailing_zeros_set(value_strings, max_precisions)
{
	_num_strings			= value_strings.length;
	_new_strings			= newArray(_num_strings);
	for(i = 0; i < _num_strings; i++)
	{
		new_value_string 	= remove_trailing_zeros(value_strings[i], max_precisions[i]);
		_new_strings[i]		= new_value_string;
	}
	return _new_strings;
}
function results_table_2_CSV(directory, filename, overwrite_existing, print_path, clear_results)
{
	filepath 			= construct_valid_file_path(directory, filename);	
	if(print_path)
		print			("------->Writing contents of results table to CSV at:\n" + filepath);
	if(!File.exists(filepath) || overwrite_existing)
	{
		saveAs			("Measurements", filepath);	
		OUTPUT_FILE_LIST	= Array.concat(OUTPUT_FILE_LIST, filepath);			
	}
	if(clear_results)
		run				("Clear Results");	
}
function ROI_material_names_2_RSPs(_ROI_material_names, _is_simulated_scan)
{
	_num_ROIs_2_analyze 		= _ROI_material_names.length;
	_ROI_material_RSPs 			= newArray(_num_ROIs_2_analyze);
	for(ROI = 0; ROI < _num_ROIs_2_analyze; ROI++)
		_ROI_material_RSPs[ROI] 	= material_name_2_RSP(_ROI_material_names[ROI], _is_simulated_scan);
	return _ROI_material_RSPs;
}
function ROI_parameter_string_2_values(_ROI_definitions_parameter_string, _ROI_definitions_parameter_list, _ROI_parameter_strings, _ROI_parameter_decodings, _ROI_parameter_value_parseFloat)
{
	for(i = 0; i < _ROI_definitions_parameter_list.length; i++)
	{	
		if( matches(ROI_definitions_parameter_list[i], _ROI_definitions_parameter_string))
			_ROI_parameter_string_index			= i;			
	}
	_desired_ROI_parameter_string 				= _ROI_parameter_strings[_ROI_parameter_string_index];
	_desired_ROI_parameter_string_elements 		= split(_desired_ROI_parameter_string, ",");
	_num_desired_ROI_parameter_string_elements 	= _desired_ROI_parameter_string_elements.length;
	_desired_ROI_parameter_values				= newArray(_num_desired_ROI_parameter_string_elements);
	for(i = 0; i < _num_desired_ROI_parameter_string_elements; i++)
	{
		_spaceless_parameter_value_string 		= strip_surrounding_spaces(_desired_ROI_parameter_string_elements[i]);
		if(_ROI_parameter_decodings[_ROI_parameter_string_index] == _ROI_parameter_value_parseFloat)
			_desired_ROI_parameter_values[i] 	= parseFloat(_spaceless_parameter_value_string);
		else
			_desired_ROI_parameter_values[i] 	= _spaceless_parameter_value_string;
	}
	if( _num_desired_ROI_parameter_string_elements == 1)
		return _desired_ROI_parameter_values[0];
	else
		 return _desired_ROI_parameter_values;
}
function save_AVI(directory, filename, compression, frame_rate, overwrite_existing, print_path, close_stack_after_saving)
{
	path 					= construct_valid_file_path(directory, filename);	
	if(print_path)
		print				("------->Saving image stack as AVI and writing to:\n" + path);
	if(!File.exists(path) || overwrite_existing)
	{
		run					("AVI... ", "compression=" + compression + " frame=" + frame_rate + " save=" + path);			
		OUTPUT_FILE_LIST	= Array.concat(OUTPUT_FILE_LIST, path);		
	}
	if(close_stack_after_saving)
		run					("Close All");
}
function save_GIF(directory, filename, overwrite_existing, print_path, _close_stack_after_saving)
{
	path 		= construct_valid_file_path(directory, filename);	
	if(print_path)
		print	("------->Saving image stack as GIF and writing to:\n" + path);
	if(!File.exists(path) || overwrite_existing)
	{
		run("Animated Gif ... ", "number=0 filename=" + path);
		OUTPUT_FILE_LIST	= Array.concat(OUTPUT_FILE_LIST, path);	
	}	
	if(_close_stack_after_saving)
		run					("Close All");
}
function save_PNG(directory, filename, overwrite_existing, close_image, print_path)
{
	PNG_filename 			= construct_valid_file_path(directory, filename);		
	if(print_path)
		print				("------->Saving plot as PNG image and writing to:\n" + PNG_filename);
	if(!File.exists(PNG_filename) || overwrite_existing)
	{
		saveAs				("PNG", PNG_filename);			
		OUTPUT_FILE_LIST	= Array.concat(OUTPUT_FILE_LIST, PNG_filename);		
	}
	if(close_image)
		close				();	
	return PNG_filename;	
}
function series_product(series)
{
	product 		= 1;
	series_length 	= series.length;
	for(i = 0; i < series_length; i++)
		product 	*= series[i];
	return product;
}
function series_partial_product(series, start_index, end_index)
{
	partial_product 	= 1;
	series_length 		= series.length;
	for(i = start_index; i <= end_index; i++)
		partial_product	*= series[i];
	return partial_product;
}
function series_sum(series)
{
	sum 			= 0;
	series_length 	= series.length;
	for(i = 0; i < series_length; i++)
		sum 		+= series[i];
	return sum;
}
function series_partial_sum(series, start_index, end_index)
{
	partial_sum 	= 0;
	series_length 	= series.length;
	for(i = start_index; i <= end_index; i++)
		partial_sum	+= series[i];
	return partial_sum;
}
function set_plot_data_props(_data_line_color, _axis_label_text_size, _x_frame_size, _y_frame_size)
{	
	Plot.setColor			(_data_line_color);				
	Plot.setAxisLabelSize	(_axis_label_text_size);
	Plot.setFormatFlags		("11111111111111");	
	Plot.setFrameSize		(_x_frame_size, _y_frame_size);
	Plot.setLimitsToFit		();									
	Plot.show				();										
}
function set_plot_extrema(_plot_data, _additional_data, _tolerance, _lower_limits_scale, _upper_limits_scale, _difference_scale)
{
	_plot_extrema 				= newArray(2);
	_extrema 					= find_array_extrema(_plot_data, _tolerance, true);
	//_lower_limits_scale = 1.0;
	//_upper_limits_scale = 1.0;
	if(_additional_data.length != 0)
	{
		_additional_data_extrema	= find_array_extrema(_additional_data, _tolerance, false);
		_plot_extrema[0] 		= minOf(_extrema[0], _additional_data_extrema[0]);
		_plot_extrema[1] 		= maxOf(_extrema[1], _additional_data_extrema[1]) ;
		//plot_extrema[0] 		= minOf(extrema[0], additional_data_extrema[0]) * lower_limits_scale;
		//plot_extrema[1] 		= maxOf(extrema[1], additional_data_extrema[1]) * upper_limits_scale;
	}
	else
	{
		_plot_extrema[0] 		= _extrema[0];
		_plot_extrema[1] 		= _extrema[1];
		//plot_extrema[0] 		= extrema[0] * lower_limits_scale;
		//plot_extrema[1] 		= extrema[1] * upper_limits_scale;
	}
	_diff_adjustment 			= (_plot_extrema[1] - _plot_extrema[0]) * _difference_scale;
	_plot_extrema[0] 			*= _lower_limits_scale;
	_plot_extrema[1] 			*= _upper_limits_scale;
	//_plot_extrema[0] 			-= _diff_adjustment;
	//_plot_extrema[1] 			+= _diff_adjustment;
	
	if(_plot_extrema[0] < 0)
		_plot_extrema[0] 		*= _upper_limits_scale / _lower_limits_scale;
	if(_plot_extrema[1] < 0)
		_plot_extrema[1] 		*= _lower_limits_scale / _upper_limits_scale;
	return _plot_extrema;
}
function set_plot_props(_legend_entries, _legend_text_size, _x_frame_size, _y_frame_size, _text, _text_size, _text_xpos_ratio, _text_ypos_ratio, _add_text_justification, _add_line_color, _linewidth)
{
	//run					("Profile Plot Options...", "width=1000 height=1000 font=20 minimum=0 maximum=0 draw draw_ticks interpolate sub-pixel");
	_legend_text 			= _legend_entries[0];
	for(i = 1; i < _legend_entries.length; i++)
		_legend_text 		+= "\n" + _legend_entries[i];
	Plot.setFontSize		(_legend_text_size);
	Plot.addLegend			(_legend_text, "Bottom-Right");						
	Plot.setLineWidth		(_linewidth);
	setForegroundColor		(255, 255, 255);
	//run					("Colors...", "foreground=white background=white selection=yellow");	
	//run					("Canvas Size...", "width=900 height=500 position=Center");	
	Plot.setJustification	(_add_text_justification);	
	Plot.setFontSize		(_text_size);
	Plot.setFormatFlags		("11111111111111");	
	Plot.addText			(_text, _text_xpos_ratio, _text_ypos_ratio);							
	Plot.setColor			(_add_line_color);		
	//run						("Size...", "width=" + _x_frame_size + " height=" + _y_frame_size + " constrain average interpolation=Bilinear");															
}
function set_ROI_y_coordinates(slice, rows_per_slice, y_offsets)
{
	y_upper_LHS 			= (slice - 1) * rows_per_slice;			// y-coordinate of upper left corner of slice to be analyzed
	y_coordinates 			= Array.copy(y_offsets);
	for (i = 0; i < y_offsets.length; i++)
		y_coordinates[i] 	+= y_upper_LHS;
	return y_coordinates;
}
function slice_2_PNG(directory, filename, slice, flip_horizontally, flip_vertically, min, max, overwrite_existing, window_title, close_image )
{					
	print				("-------> Extracting and saving PNG image of the slice of x being analysed...");			
	dimensions 			= query_open_image_dimensions();
	columns 			= dimensions[0];
	rows  				= dimensions[1];					
	image_slices 		= dimensions[2]; 					
	x_colunns 			= x_magnification * columns;
	x_rows 				= x_magnification * rows;
	y_upper_LHS 		= (slice - 1) * rows;			// y-coordinate of upper left corner of slice to be analyzed
	makeRectangle		( 0, y_upper_LHS, columns, rows);
	//imageCalculator	("Copy create", filename ,filename );
	//run				("Enhance Contrast", "saturated=0.35");
	run					("Duplicate...", "title=" + window_title );
	setMinAndMax		(min, max);
	if(flip_horizontally)
		run				("Flip Horizontally");
	if(flip_vertically)
		run				("Flip Vertically");
	run					("Size...", "width=x_columns height=x_rows constrain average interpolation=Bilinear");
	save_PNG			(directory, filename, overwrite_existing, close_image, false);				
}
function stack_2_animated_GIF(_stack_title, _GIF_directory, _GIF_filename, _GIF_speed, _GIF_2_disk, _overwrite_GIF, _print_GIF_path, _close_stack_after_saving)
{
	selectImage		(_stack_title);
	run				("Animation Options...", "speed=" + _GIF_speed);
	if(_GIF_2_disk)
		save_GIF	(_GIF_directory, _GIF_filename, _overwrite_GIF, _print_GIF_path, _close_stack_after_saving);
	
}
function stack_2_AVI(stack_title, directory, filename, compression, frame_rate, AVI_2_disk, overwrite_AVI, print_path, close_stack_after_saving)
{
	selectImage		(stack_title);
	if(AVI_2_disk)
		save_AVI	(directory, filename, compression, frame_rate, overwrite_AVI, print_path, close_stack_after_saving);
}
function string_2_array(string) { return split(string); }
function string_array_concatenation(_string_array, _front_concatenation_string, _back_concatenation_string)
{
	_num_array_strings = _string_array.length;
	_concatenated_string_array = newArray(_num_array_strings);
	for(i = 0; i < _num_array_strings; i++)
		_concatenated_string_array[i] = _front_concatenation_string + _string_array[i] + _back_concatenation_string;
	return _concatenated_string_array;
}
function strip_surrounding_spaces(string)
{
	spaceless_string 			= string;
	do 
	{
		starts_with_space 		= startsWith(spaceless_string, " ") || startsWith(spaceless_string, "\n") || startsWith(spaceless_string, "\t");
		if(starts_with_space)
			spaceless_string 	= substring(spaceless_string, 1);
	} 
	while (starts_with_space);
	do 
	{
		string_length			= lengthOf(spaceless_string); 
		ends_with_space 		= endsWith(spaceless_string, " ") || endsWith(spaceless_string, "\n") || endsWith(spaceless_string, "\t");
		if(ends_with_space)
			spaceless_string 	= substring(spaceless_string, 0, string_length - 1);
	} 
	while (ends_with_space);	
	return spaceless_string;
}
function subdirectory_array_concatenation(_subdirectory_array, _front_concatenation_subdirectory, _back_concatenation_subdirectory)
{
	_num_array_subdirectories = _subdirectory_array.length;
	_concatenated_subdirectory_array = newArray(_num_array_subdirectories);
	for(i = 0; i < _num_array_subdirectories; i++)
	{
		if(lengthOf(_front_concatenation_subdirectory) > 0)
		{	
			_concatenated_subdirectory = construct_valid_directory_path(_front_concatenation_subdirectory, _subdirectory_array[i]);
			//print				("_concatenated_subdirectory =" + _concatenated_subdirectory);		
			if(lengthOf(_back_concatenation_subdirectory) > 0)						
			{	
				_concatenated_subdirectory = construct_valid_directory_path(_concatenated_subdirectory, _back_concatenation_subdirectory);
				//print				("_concatenated_subdirectory =" + _concatenated_subdirectory);	
			}
		}							
		else
		{
			if(lengthOf(_back_concatenation_subdirectory) > 0)						
			{	
				_concatenated_subdirectory = construct_valid_directory_path(_subdirectory_array[i], _back_concatenation_subdirectory);
				//print				("_concatenated_subdirectory2 =" + _concatenated_subdirectory);	
			}
			else
			{
				_concatenated_subdirectory = _subdirectory_array[i];
				//print				("_concatenated_subdirectory2 =" + _concatenated_subdirectory);			
			}
		}		
		_concatenated_subdirectory_array[i]	= 		_concatenated_subdirectory;			
	}
	return _concatenated_subdirectory_array;
}
function suffixes_2_filenames(common_basename, filename_suffixes)
{
	num_suffixes		= filename_suffixes.length;
	filenames			= newArray(num_suffixes);
	for(i = 0; i < num_suffixes; i++)
		filenames[i]	= common_basename + filename_suffixes[i];
	return 				filenames;
}
function verify_distributed_files(_paths)
{
	_num_files = _paths.length;
	if(_num_files == 0)
		return false;
	for(i = 0; i < _num_files; i++)
	{
		//file_path = construct_valid_file_path(directories[i], filenames[i]);
		if(!File.exists(_paths[i]))
			return false;
	}
	return true;
}
function verify_distributed_file_set(directories, filenames)
{
	num_files = filenames.length;
	if(num_files == 0)
		return false;
	for(i = 0; i < num_files; i++)
	{
		file_path = construct_valid_file_path(directories[i], filenames[i]);
		if(!File.exists(file_path))
			return false;
	}
	return true;
}
function verify_file_set(common_directory, common_basename, filename_suffixes, file_extension)
{
	num_files = filename_suffixes.length;
	file_set_validated = true;
	for(i = 0; i < num_files; i++)
	{
		file_path = construct_valid_file_path(common_directory, common_basename + filename_suffixes[i] + file_extension);
		if(!File.exists(file_path))
			return false;
	}
	return true;
}
function verify_input_data(_common_directory, _current_parameter_value_test_folders, _current_analysis_RSP_input_data_filenames, _current_analysis_RSP_error_input_data_filenames, _current_analysis_std_dev_input_data_filenames, _current_analysis_TV_input_data_filenames)
{
	for( i = 0; i < _current_parameter_value_test_folders.length; i++)
	{
		directory 	= _common_directory + _current_parameter_value_test_folders[i];
		print		("------->Verifying RSP input data file exists at:\n" + directory + _current_analysis_RSP_input_data_filenames);
		print		("------->Verifying RSP error input data file exists at:\n" + directory + _current_analysis_RSP_error_input_data_filenames);
		print		("------->Verifying standard deviation input data file exists at:\n" + directory + _current_analysis_std_dev_input_data_filenames);
		print		("------->Verifying total variation (TV) input data file exists at:\n" + directory + _current_analysis_TV_input_data_filenames);
		if(!File.exists(directory + _current_analysis_RSP_input_data_filenames))
			return false;
		if(!File.exists(directory + _current_analysis_RSP_error_input_data_filenames));
			return false;
		if(!File.exists(directory + _current_analysis_std_dev_input_data_filenames));
			return false;
		if(!File.exists(directory + _current_analysis_TV_input_data_filenames));
			return false;
	}
	return true;						
}
function verify_paths_exist(_path_strings, _print_missing_paths)
{
	_missing_paths = newArray();
	for(i = 0; i < _path_strings.length; i++)
	{
		if(!File.exists(_path_strings[i]))
        	_missing_paths = Array.concat(_missing_paths, _path_strings[i]);		
	}
	if(_missing_paths.length > 0)
	{
		print("=======> One or more paths were not found");
	    if(_print_missing_paths)
	    	Array.print(_missing_paths);
	}
	else
		print("=======> All paths exist");
	return _missing_paths;
}
function verify_reconstruction_output(_reconstruction_data_directory, _reconstruction_output_directory, _reconstructed_image_filenames, _TV_input_data_filename, _print_path)
{
	_reconstruction_output_directory_path = construct_valid_directory_path(_reconstruction_data_directory, _reconstruction_output_directory);		
	if(_print_path)
		print				("-------> Verifying reconstruction output data exists for:\n" + _reconstruction_output_directory_path				);		
	else
		print				("-------> Verifying reconstruction output data exists...");		
	data_verified 			= true;
	_num_iterations			= _reconstructed_image_filenames.length;
	for(iteration = 0; iteration < _num_iterations; iteration++)
	{
		check_image_path 	= _reconstruction_output_directory_path + File.separator + _reconstructed_image_filenames[iteration];
		if(!File.exists(check_image_path))
		{
			print			("ERROR: reconstructed image does not exist:\n" + check_image_path);
			data_verified 	= false;
			missing_data	= Array.concat(missing_data, check_image_path);
		}			
	}
	check_TV_path			= _reconstruction_output_directory_path + File.separator + _TV_input_data_filename;
	if(!File.exists(check_TV_path))
	{
		print				("ERROR: TV measurement file does not exist:\n" + check_TV_path);
		data_verified 		= false;
		missing_data		= Array.concat(missing_data, check_TV_path);
	}
	if(data_verified)
		print		("=======> RECONSTRUCTION OUTPUT DATA VERIFIED\n");
	else
		print		("=======> RECONSTRUCTION OUTPUT DATA NOT FOUND\n");
	
	return data_verified;			
}
function verify_reconstruction_output_set(_reconstruction_data_directory, _reconstruction_output_directory, _reconstructed_image_filenames, _TV_input_data_filename)
{
	_reconstruction_output_directory_path = construct_valid_directory_path(_reconstruction_data_directory, _reconstruction_output_directory);		
	print				("-------> Verifying reconstruction output data exists for:\n" + _current_parameter_value_test_folder				);		
	//_reconstruction_output_data_
	data_verified 			= true;
	_num_iterations			= _reconstructed_image_filenames.length;
	for(iteration = 0; iteration < _num_iterations; iteration++)
	{
		check_image_path 	= _reconstruction_output_directory_path + File.separator + _reconstructed_image_filenames[iteration];
		if(!File.exists(check_image_path))
		{
			print			("ERROR: one or more reconstructed images do not exist:\n" + check_image_path);
			data_verified 	= false;
			missing_data	= Array.concat(missing_data, check_image_path);
		}			
	}
	check_TV_path			= _reconstruction_output_directory_path + File.separator + _TV_input_data_filename;
	if(!File.exists(check_TV_path))
	{
		print				("ERROR: TV measurement file does not exist:\n" + check_TV_path);
		data_verified 		= false;
		missing_data		= Array.concat(missing_data, check_TV_path);
	}
	if(data_verified)
		print		("=======> RECONSTRUCTION OUTPUT DATA VERIFIED\n" + _current_parameter_value_test_folder);
	else
		print		("=======> RECONSTRUCTION OUTPUT DATA NOT FOUND\n" + _current_parameter_value_test_folder);
	return data_verified;			
}
function verify_ROI_analysis_output_files(_common_directory, _current_parameter_value_test_folder, _ROI_analysis_RSP_output_filename, _ROI_analysis_RSP_error_output_filename, _ROI_analysis_std_dev_output_filename, _ROI_analysis_TV_input_filename, _slices_2_analyze_folders, _ROI_selection_diameter_folders, _print_path)
{
	_missing_data 		= false;
	if(_print_path)
		print				("-------> Verifying ROI analysis has been performed on the reconstruction data at:\n" + _current_parameter_value_test_folder				);		
	else
		print				("-------> Verifying ROI analysis has been performed on the reconstruction data");		
	for(_ROI_selection_diameter_index = 0; _ROI_selection_diameter_index < _ROI_selection_diameter_folders.length; _ROI_selection_diameter_index++)
	{
		for(_slice_2_analyze_index = 0; _slice_2_analyze_index < _slices_2_analyze_folders.length; _slice_2_analyze_index++)
		{
			_slice_2_analyze_folder 			= _slices_2_analyze_folders[_slice_2_analyze_index];
			_ROI_selection_diameter_folder 		= _ROI_selection_diameter_folders[_ROI_selection_diameter_index];					
			_current_analysis_subdirectory		= _slice_2_analyze_folder + _ROI_selection_diameter_folder;					
			_current_ROI_analysis_subdirectory	= _slice_2_analyze_folder + _ROI_selection_diameter_folder;					
			_current_TV_analysis_subdirectory	= "";					
				
			_current_analysis_ROI_analysis_RSP_output_filename			= _current_ROI_analysis_subdirectory 	+ File.separator 	+ _ROI_analysis_RSP_output_filename;
			_current_analysis_ROI_analysis_RSP_error_output_filename	= _current_ROI_analysis_subdirectory 	+ File.separator 	+ _ROI_analysis_RSP_error_output_filename;
			_current_analysis_ROI_analysis_std_dev_output_filename		= _current_ROI_analysis_subdirectory 	+ File.separator 	+ _ROI_analysis_std_dev_output_filename;
			_current_analysis_ROI_analysis_TV_input_filename			= _current_TV_analysis_subdirectory 	+ File.separator	+ _ROI_analysis_TV_input_filename;				
			
			_source_folder		= construct_valid_directory_path(_common_directory, _current_parameter_value_test_folder				);
			_RSP_file_path		= construct_valid_file_path(_source_folder, _current_analysis_ROI_analysis_RSP_output_filename			);
			_RSP_error_file_path= construct_valid_file_path(_source_folder, _current_analysis_ROI_analysis_RSP_error_output_filename	);
			_std_dev_file_path	= construct_valid_file_path(_source_folder, _current_analysis_ROI_analysis_std_dev_output_filename		);
			_TV_file_path		= construct_valid_file_path(_source_folder, _current_analysis_ROI_analysis_TV_input_filename			);

			//print("RSP file path = " + _RSP_file_path);
			//print("RSP error file path = " + _RSP_error_file_path);
			//print("RSP std dev file path = " + _std_dev_file_path);
			//print("TV file path = " + _TV_file_path);
			if(!File.exists(_RSP_file_path))
				_missing_data = true;
			if(!File.exists(_RSP_error_file_path))
				_missing_data = true;
			if(!File.exists(_std_dev_file_path))
				_missing_data = true;
			if(!File.exists(_TV_file_path))
				_missing_data = true;				
		}
	}
	if(_missing_data)
		print		("=======> ROI analysis data missing for:\n" + _current_parameter_value_test_folder);
	else
		print		("=======> ROI ANALYSIS HAS BEEN COMPLETED\n" + _current_parameter_value_test_folder);
	return _missing_data;						
}
function verify_ROI_analysis_output_file_set(_common_directory, _current_parameter_value_test_folders, _current_analysis_RSP_input_data_filenames, _current_analysis_RSP_error_input_data_filenames, _current_analysis_std_dev_input_data_filenames, _current_analysis_TV_input_data_filenames)
{
	missing_data_directories = newArray();
	print_section_separator		("------->Verifying ROI analysis output data files exist for specified set of reconstructions:\n", PRINT_MAJOR_SECTION);
	for( i = 0; i < _current_parameter_value_test_folders.length; i++)
	{
		missing_data = false;
		directory 	= _common_directory + _current_parameter_value_test_folders[i];
		if(!File.exists(directory + _current_analysis_RSP_input_data_filenames))
			missing_data = true;
		if(!File.exists(directory + _current_analysis_RSP_error_input_data_filenames))
			missing_data = true;
		if(!File.exists(directory + _current_analysis_std_dev_input_data_filenames))
			missing_data = true;
		if(!File.exists(directory + _current_analysis_TV_input_data_filenames))
			missing_data = true;
		if(missing_data)
			missing_data_directories = Array.concat(missing_data_directories, directory);		
	}
	if(missing_data_directories.length > 0)
	{
		print		("=======> ROI analysis data missing for:\n", PRINT_MINOR_SECTION);
		for(i=0; i < missing_data_directories.length; i++)
			print("-------> " + missing_data_directories[i]);
	}
	return missing_data_directories;						
}
