#include <fdmt_gpu.hpp>
FDMTGPU::FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps,
    float tsamp, size_t dt_max, size_t dt_step, size_t dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min)
{
    // Allocate memory for the state buffers
    const auto& plan = get_plan();
    transfer_plan_to_device();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    m_state_in_d.resize(state_size, 0.0F);
    m_state_out_d.resize(state_size, 0.0F);
}
FDMTPlanD FDMTGPU::transfer_plan_to_device()
{
    const auto& plan = get_plan();
    //FDMTPlanD plan_d;
    //for (const auto& state_shape_iter : plan.state_shape) {
    //    for (const auto& shape : state_shape_iter) {
    //        plan_d.state_shape_d.push_back(shape);
    //    }
    //}
    //// flatten sub_plan and transfer to device
    //for (const auto& sub_plan_iter : plan.sub_plan) {
    //    for (const auto& sub_plan : sub_plan_iter) {
    //        plan_d.state_idx_d.push_back(sub_plan.state_idx);
    //        for (const auto& dt : sub_plan.dt_grid) {
    //            plan_d.dt_grid_d.push_back(dt);
    //        }
    //        for (const auto& dt_tuple : sub_plan.dt_plan) {
    //            for (const auto& idt : dt_tuple) {
    //                plan_d.dt_plan_d.push_back(idt);
    //            }
    //        }
    //    }
    //}
    //return plan_d;
    return FDMTPlanD();
};