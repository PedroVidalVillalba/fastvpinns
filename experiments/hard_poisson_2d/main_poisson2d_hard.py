import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import yaml
import sys
import copy
from tensorflow.keras import layers
from tensorflow.keras import initializers
from rich.console import Console
import copy
import time


from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D
from fastvpinns.model.model_hard import DenseModel_Hard
from fastvpinns.physics.poisson2d import pde_loss_poisson
from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table

# import the example file
from force_and_exact import *

# import all files from utility
from utility import *

if __name__ == "__main__":

    console = Console()

    # check input arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <input file>")
        sys.exit(1)

    # Read the YAML file
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Extract the values from the YAML file
    i_output_path = config['experimentation']['output_path']

    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    i_generate_mesh_plot = config['geometry']['generate_mesh_plot']
    i_mesh_type = config['geometry']['mesh_type']
    i_x_min = config['geometry']['internal_mesh_params']['x_min']
    i_x_max = config['geometry']['internal_mesh_params']['x_max']
    i_y_min = config['geometry']['internal_mesh_params']['y_min']
    i_y_max = config['geometry']['internal_mesh_params']['y_max']
    i_n_cells_x = config['geometry']['internal_mesh_params']['n_cells_x']
    i_n_cells_y = config['geometry']['internal_mesh_params']['n_cells_y']
    i_n_test_points_x = config['geometry']['internal_mesh_params']['n_test_points_x']
    i_n_test_points_y = config['geometry']['internal_mesh_params']['n_test_points_y']

    i_fe_order = config['fe']['fe_order']
    i_fe_type = config['fe']['fe_type']
    i_quad_order = config['fe']['quad_order']
    i_quad_type = config['fe']['quad_type']

    i_model_architecture = config['model']['model_architecture']
    i_activation = config['model']['activation']
    i_use_attention = config['model']['use_attention']
    i_epochs = config['model']['epochs']
    i_dtype = config['model']['dtype']
    if i_dtype == "float64":
        i_dtype = tf.float64
    elif i_dtype == "float32":
        i_dtype = tf.float32
    else:
        print("[ERROR] The given dtype is not a valid tensorflow dtype")
        raise ValueError("The given dtype is not a valid tensorflow dtype")

    i_set_memory_growth = config['model']['set_memory_growth']
    i_learning_rate_dict = config['model']['learning_rate']

    i_update_console_output = config['logging']['update_console_output']

    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # Initiate a Geometry_2D object
    domain = Geometry_2D(
        i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path
    )

    # FEspace expects boundary conditions (set trivial)
    zero_fn = lambda x, y : 0.0
    bound_function_dict = {1000: zero_fn, 1001: zero_fn, 1002: zero_fn, 1003: zero_fn}
    bound_condition_dict = {1000: "dirichlet", 1001: "dirichlet", 1002: "dirichlet", 1003: "dirichlet"}

    # load the mesh
    with open('results.txt', 'w') as out:
        print("p,q,n,h,Epoch,L2 error,L1 error,Linf error,L2 relative,L1 relative,Linf relative,Loss", file=out, flush=True)
        for p in range(2, 9):
            for q in range(3, 9):
                for n in range(1, 33):
                    h = 1.0 / n
                    cells, boundary_points = domain.generate_quad_mesh_internal(
                        x_limits=[i_x_min, i_x_max],
                        y_limits=[i_y_min, i_y_max],
                        n_cells_x=n,
                        n_cells_y=n,
                        num_boundary_points=4,  # Domain expects some boundary points (set to minimum (1 on each side))
                    )


                    fespace = Fespace2D(
                        mesh=domain.mesh,
                        cells=cells,
                        boundary_points=boundary_points,
                        cell_type=domain.mesh_type,
                        fe_order=p,
                        fe_type=i_fe_type,
                        quad_order=q,
                        quad_type=i_quad_type,
                        fe_transformation_type="bilinear",
                        bound_function_dict=bound_function_dict ,
                        bound_condition_dict=bound_condition_dict,
                        forcing_function=rhs,
                        output_path=i_output_path,
                        generate_mesh_plot=i_generate_mesh_plot,
                    )

                    test_points = domain.get_test_points()
                    # print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
                    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

                    # # save points for plotting
                    # X = test_points[:, 0].reshape(i_n_test_points_x, i_n_test_points_y)
                    # Y = test_points[:, 1].reshape(i_n_test_points_x, i_n_test_points_y)
                    # Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)

                    # plot the exact solution
                    # plot_contour(
                    #     x=X,
                    #     y=Y,
                    #     z=Y_Exact_Matrix,
                    #     output_path=i_output_path,
                    #     filename="exact_solution",
                    #     title="Exact Solution",
                    # )

                    # instantiate data handler
                    datahandler = DataHandler2D(fespace, domain, dtype=i_dtype)

                    params_dict = {}
                    params_dict['n_cells'] = fespace.n_cells

                    # get bilinear parameters
                    # this function will obtain the values of the bilinear parameters from the model
                    # and convert them into tensors of desired dtype
                    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

                    for i in range(10):
                        model = DenseModel_Hard(
                            layer_dims=i_model_architecture,
                            learning_rate_dict=i_learning_rate_dict,
                            params_dict=params_dict,
                            loss_function=pde_loss_poisson,
                            input_tensors_list=[datahandler.x_pde_list],
                            orig_factor_matrices=[
                                datahandler.shape_val_mat_list,
                                datahandler.grad_x_mat_list,
                                datahandler.grad_y_mat_list,
                            ],
                            force_function_list=datahandler.forcing_function_list,
                            tensor_dtype=i_dtype,
                            use_attention=i_use_attention,
                            activation=i_activation,
                            hessian=False,
                            hard_constraint_function=apply_hard_boundary_constraints,
                        )

                        num_epochs = i_epochs  # num_epochs
                        progress_bar = tqdm(
                            total=num_epochs,
                            desc='Training',
                            unit='epoch',
                            bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
                            colour="green",
                            ncols=100,
                        )
                        loss_array = []  # total loss
                        test_loss_array = []  # test loss
                        time_array = []  # time per epoc

                        # ---------------------------------------------------------------#
                        # ------------- TRAINING LOOP ---------------------------------- #
                        # ---------------------------------------------------------------#
                        for epoch in range(num_epochs):

                            # Train the model
                            batch_start_time = time.time()
                            loss = model.train_step(bilinear_params_dict=bilinear_params_dict)
                            elapsed = time.time() - batch_start_time

                            # print(elapsed)
                            time_array.append(elapsed)

                            loss_array.append(loss['loss'])

                            # ------ Intermediate results update ------ #
                            if (epoch + 1) % i_update_console_output == 0 or epoch == num_epochs - 1:
                                y_pred = model(test_points).numpy()
                                y_pred = y_pred.reshape(-1)

                                error = np.abs(y_exact - y_pred)

                                # get errors
                                (
                                    l2_error,
                                    linf_error,
                                    l2_error_relative,
                                    linf_error_relative,
                                    l1_error,
                                    l1_error_relative,
                                ) = compute_errors_combined(y_exact, y_pred)

                                loss_pde = float(loss['loss_pde'].numpy())
                                loss_dirichlet = float(loss['loss_dirichlet'].numpy())
                                total_loss = float(loss['loss'].numpy())

                                # Append test loss
                                test_loss_array.append(l1_error)


                                print(f"{p:2},{q:2},{n:2},{h:2},{epoch + 1:5},{l2_error:.6e},{l1_error:.6e},{linf_error:.6e},"
                                      f"{l2_error_relative:.6e},{l1_error_relative:.6e},{linf_error_relative:.6e},{total_loss:.6e}", file=out, flush=True)

                                console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
                                console.print("[bold]--------------------[/bold]")
                                console.print(
                                    f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]"
                                )
                                console.print(
                                    f"Test Losses        || L1 Error : {l1_error:.3e} L2 Error : {l2_error:.3e} Linf Error : {linf_error:.3e}"
                                )

                                # plot_results(
                                #     loss_array,
                                #     test_loss_array,
                                #     y_pred,
                                #     X,
                                #     Y,
                                #     Y_Exact_Matrix,
                                #     i_output_path,
                                #     epoch,
                                #     i_n_test_points_x,
                                #     i_n_test_points_y,
                                # )

                            progress_bar.update(1)
                        # print(f"{p:2}, {q:2}, {h:2}, {l2_error:.6e}", file=out, flush=True)


                        # Save the model
                        # model.save_weights(str(Path(i_output_path) / "model_weights"))

                        # print the Error values in table
                        print_table(
                            "Error Values",
                            ["Error Type", "Value"],
                            [
                                "L2 Error",
                                "Linf Error",
                                "Relative L2 Error",
                                "Relative Linf Error",
                                "L1 Error",
                                "Relative L1 Error",
                            ],
                            [l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative],
                        )

                        # print the time values in table
                        print_table(
                            "Time Values",
                            ["Time Type", "Value"],
                            [
                                "Time per Epoch(s) - Median",
                                "Time per Epoch(s) IQR-25% ",
                                "Time per Epoch(s) IQR-75% ",
                                "Mean (s)",
                                "Epochs per second",
                                "Total Train Time",
                            ],
                            [
                                np.median(time_array),
                                np.percentile(time_array, 25),
                                np.percentile(time_array, 75),
                                np.mean(time_array),
                                int(i_epochs / np.sum(time_array)),
                                np.sum(time_array),
                            ],
                        )

                    # save all the arrays as numpy arrays
                    # np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
                    # np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
                    # np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
                    # np.savetxt(str(Path(i_output_path) / "error.txt"), error)
                    # np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))
