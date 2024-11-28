import matplotlib.pyplot as plt
import numpy as np

def visualize_pw_bounds(i, j, parameter_samples, image_samples, safe_lin_lb, safe_lin_ub, safe_pw_bounds, pw_indicator):
    plt.figure()

    # Image ---------------------------------------------------
    plt.subplot(1, 2, 1)
    plt.title("Image")
    sample_idx = 0
    plt.imshow(image_samples[0, :, :, 0, sample_idx], cmap="gray")
    plt.plot(j, i, "ro")

    # Pixel values --------------------------------------------
    plt.subplot(1, 2, 2)
    plt.title("Pixel Values")
    kappas = parameter_samples[0, 0]
    pixel_values = image_samples[0, i, j, 0]
    plt.plot(kappas, pixel_values, "red", lw=2, label="Pixel values")

    # Linear bounds -------------------------------------------
    kappas = parameter_samples[0, 0]
    w, b = safe_lin_lb[i, j][0]
    lb_values = w * kappas + b
    plt.plot(kappas, lb_values, color="gray", linestyle="--", label="Linear bound")

    w, b = safe_lin_ub[i, j][0]
    ub_values = w * kappas + b
    plt.plot(kappas, ub_values, color="gray", linestyle="--")
    plt.fill_between(kappas, lb_values, ub_values, color="gray", alpha=0.5)

    # PWL bound ----------------------------------------------
    is_lower_valid = pw_indicator['lower'][i, j]
    if is_lower_valid:
        w_lo_1, b_lo_1 = safe_pw_bounds['lower'][i, j][0]
        w_lo_2, b_lo_2 = safe_lin_lb[i, j][0]
        lb_1 = w_lo_1 * kappas + b_lo_1
        lb_2 = w_lo_2 * kappas + b_lo_2
        lb = np.maximum(lb_1, lb_2)

        w_hi, b_hi = safe_lin_ub[i, j][0]
        ub = w_hi * kappas + b_hi
    else:
        w_lo, b_lo = safe_lin_lb[i, j][0]
        lb = w_lo * kappas + b_lo

        w_hi_1, b_hi_1 = safe_pw_bounds['upper'][i, j][0]
        w_hi_2, b_hi_2 = safe_lin_ub[i, j][0]
        ub_1 = w_hi_1 * kappas + b_hi_1
        ub_2 = w_hi_2 * kappas + b_hi_2
        ub = np.minimum(ub_1, ub_2)

    plt.plot(kappas, lb, color="blue", label="PWL bound")
    plt.plot(kappas, ub, color="blue")
    plt.fill_between(kappas, lb, ub, color="blue", alpha=0.5)

    # Finish figure -------------------------------------------
    plt.legend()
    plt.grid()
    plt.xlabel("Parameter")
    plt.ylabel("Pixel Values")
    plt.suptitle(f"Pixel ({i}, {j})")
    plt.tight_layout()

    plt.show()
