import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

rolling_window = 100


def plot_results(result_dir, disc_losses, gen_losses, w_distances, gradient_penalty_list):
    disc_losses = [-x for x in disc_losses]
    fig = plt.figure()
    plt.title('Critic Negative Loss')
    smoothed = pd.DataFrame(disc_losses).ewm(alpha=0.1, adjust=False)
    plt.plot(range(len(disc_losses)), disc_losses, alpha=0.7)
    plt.plot(range(len(disc_losses)), smoothed.mean()[0])
    plt.xlabel('Training steps')
    if args.log:
        plt.yscale('log')
    plt.ylabel('Loss')
    plt.savefig('{}discriminator_loss_smoothed'.format(result_dir), dpi=300)
    plt.close(fig)

    fig = plt.figure()
    plt.title('Generator Loss')
    smoothed = pd.DataFrame(gen_losses).ewm(alpha=0.1, adjust=False)
    plt.plot(range(len(gen_losses)), gen_losses, alpha=0.7)
    plt.plot(range(len(gen_losses)), smoothed.mean()[0])
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.savefig('{}generator_loss_smoothed'.format(result_dir), dpi=300)
    plt.close(fig)

    fig = plt.figure()
    plt.title('Wasserstein Distance Estimate')
    smoothed = pd.DataFrame(w_distances).ewm(alpha=0.1, adjust=False)
    plt.plot(range(len(w_distances)), w_distances, alpha=0.7)
    plt.plot(range(len(w_distances)), smoothed.mean()[0])
    plt.xlabel('Training steps')
    if args.log:
        plt.yscale('log')
    plt.ylabel('Distance')
    plt.savefig('{}wasserstein_distance'.format(result_dir), dpi=300)
    plt.close(fig)

    fig = plt.figure()
    plt.title('Gradient Penalty')
    smoothed = pd.DataFrame(gradient_penalty_list).ewm(alpha=0.1, adjust=False)
    plt.plot(range(len(gradient_penalty_list)), gradient_penalty_list, alpha=0.7)
    plt.plot(range(len(gradient_penalty_list)), smoothed.mean()[0])
    plt.xlabel('Training steps')
    if args.log:
        plt.yscale('log')
    plt.ylabel('Penalty')
    plt.savefig('{}gradient_penalty'.format(result_dir), dpi=300)
    plt.close(fig)


def load_list(filename):
    df = pd.read_csv(filename)
    return df['Value']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--compute_distance', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=True)
    args = parser.parse_args()

    result_dir = args.result_dir

    disc_losses = load_list('{}run_.-tag-data_D_loss.csv'.format(result_dir))
    gen_losses = load_list('{}run_.-tag-data_G_loss.csv'.format(result_dir))
    gradient_penalty_list = load_list('{}run_.-tag-data_gradient_penalty.csv'.format(result_dir))
    if args.compute_distance:
        w_distances = - disc_losses - gradient_penalty_list
    else:
        w_distances = load_list('{}run_.-tag-data_Wasserstein_distance_estimate.csv'.format(result_dir))

    '''if args.log:
        disc_losses = np.log(disc_losses)
        w_distances = np.log(w_distances)
        gradient_penalty_list = np.log(gradient_penalty_list)'''

    plot_results(
        result_dir,
        list(disc_losses),
        list(gen_losses),
        list(w_distances),
        list(gradient_penalty_list)
    )