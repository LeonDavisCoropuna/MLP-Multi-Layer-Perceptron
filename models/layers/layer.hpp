#include <vector>
class Layer
{
public:
  virtual std::vector<float> forward(const std::vector<float> &input) = 0;
  virtual void backward(const std::vector<float> *targets = nullptr,
                        const Layer *next_layer = nullptr) = 0;
  virtual void update_weights() = 0;
  virtual void accumulate_gradients() = 0;
  virtual void apply_gradients(float batch_size) = 0;
  virtual void zero_grad() = 0;

  virtual const std::vector<float> &get_outputs() const = 0;
  virtual const std::vector<std::vector<float>> &get_weights() const = 0;
  virtual const std::vector<float> &get_deltas() const = 0;
  virtual int input_size() const = 0;
  virtual int output_size() const = 0;
  virtual ~Layer() = default;
};
