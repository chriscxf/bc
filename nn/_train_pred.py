
import torch

from _basicClasses import TimeSeriesDatasetSingleSubject

from _functions import joint_likelihood

def split_dataset(dataset, r=0.4, input_window_size = 24, output_window_size = 24):
    total_time_points = len(dataset)
    split_index = int(total_time_points * (1 - r))

    train_X = dataset.X[:split_index]
    train_Y = dataset.Y[:split_index]
    # train_true_X = dataset.trueX[:split_index]
    # train_true_Y = dataset.trueY[:split_index]
    train_cat_vars = dataset.cat_vars[:split_index]
    train_binary_event = dataset.binary_event[:split_index]

    test_X = dataset.X[split_index:]
    test_Y = dataset.Y[split_index:]
    # test_true_X = dataset.trueX[split_index:]
    # test_true_Y = dataset.trueY[split_index:]
    test_cat_vars = dataset.cat_vars[split_index:]
    test_binary_event = dataset.binary_event[split_index:]

    train_dataset = TimeSeriesDatasetSingleSubject(train_X, train_Y, train_cat_vars, train_binary_event, input_window_size, output_window_size)
    test_dataset = TimeSeriesDatasetSingleSubject(test_X, test_Y, test_cat_vars, test_binary_event,input_window_size, output_window_size)

    return train_dataset, test_dataset

def train(global_model, local_model, optimizer, train_dataloader):
    global_model.train()
    local_model.train()
    # event_model.train()
    
    total_loss = 0

    for batch  in train_dataloader:
        X, Y, trueX, trueY, cat_vars, binary_events, true_binary_events = batch
        optimizer.zero_grad()

        mu = global_model(X)
        
        sigma, p, local_mu = local_model(torch.cat((Y.unsqueeze(-1), cat_vars), dim=-1))

        # p = event_model(torch.cat((Y.unsqueeze(-1), cat_vars), dim=-1))
        # print(true_binary_events.shape, Y.shape, mu.shape, sigma.shape, p.shape)
        # print(mu)

        # loss = joint_likelihood(true_binary_events, trueY, mu + local_mu, sigma, p)
        loss = joint_likelihood(true_binary_events, trueY, mu , sigma, p)
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1)
        # torch.nn.utils.clip_grad_norm_(event_model.parameters(), max_norm=1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def evaluate(global_model, local_model, test_dataloader):
    global_model.eval()  # Set the global model to evaluation mode
    local_model.eval()  # Set the local model to evaluation mode
    # event_model.eval()
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        total_loss = 0
        for X, Y, trueX, trueY,cat_vars, binary_events, true_binary_events in test_dataloader:
            mu = global_model(X)
            sigma, p, local_mu = local_model(torch.cat((Y.unsqueeze(-1), cat_vars), dim=-1))

            # loss = joint_likelihood(true_binary_events, trueY, mu+local_mu, sigma, p)
            loss = joint_likelihood(true_binary_events, trueY, mu, sigma, p)

            total_loss += loss.item()
    
    return total_loss / len(test_dataloader)



def predict(global_model, local_model,  dataloader):
    global_model.eval()
    local_model.eval()
    # event_model.eval()
    
    # Initialize lists to store outputs
    mus = []
    sigmas = []
    true_values = []
    binary_events_collected = []
    p_collected = []
    z_collected = []  # Store subject IDs

    with torch.no_grad():
        for X, Y, trueX, trueY, cat_vars, binary_events, true_binary_events in dataloader:
            mu = global_model(X)
            sigma, p, local_mu = local_model(torch.cat((Y.unsqueeze(-1), cat_vars), dim=-1))
            # z = mu + local_mu
            z = mu

            mus.append(mu)
            sigmas.append(sigma)
            true_values.append(trueY)
            binary_events_collected.append(true_binary_events)
            z_collected.append(z)  # Append subject IDs
            p_collected.append(p)
    
    # Concatenate lists to tensors
    mus = torch.cat(mus, dim=0)
    sigmas = torch.cat(sigmas, dim=0)
    true_values = torch.cat(true_values, dim=0)
    binary_events_collected = torch.cat(binary_events_collected, dim=0)
    z_collected = torch.cat(z_collected, dim=0)  
    p_collected = torch.cat(p_collected)
    
    return mus, sigmas, p_collected, z_collected, true_values, binary_events_collected

